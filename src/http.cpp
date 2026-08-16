#include "http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// Read up to n bytes (or until EOF/error). Returns bytes read, -1 on error.
ssize_t read_full(int fd, char * buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd, buf + got, n - got, 0);
        if (r > 0) { got += (size_t) r; continue; }
        if (r == 0) return (ssize_t) got;
        if (errno == EINTR) continue;
        return -1;
    }
    return (ssize_t) got;
}

} // namespace

HttpServer::HttpServer(int port, int backlog) : port_(port), backlog_(backlog) {}

HttpServer::~HttpServer() {
    if (listen_fd_ >= 0) ::close(listen_fd_);
}

void HttpServer::handle(const std::string & method, const std::string & path, Handler h) {
    routes_.emplace_back(method, path, std::move(h));
}

bool HttpServer::start() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        fprintf(stderr, "http: socket() failed: %s\n", strerror(errno));
        return false;
    }
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t) port_);
    if (::bind(listen_fd_, (sockaddr *) &addr, sizeof addr) != 0) {
        fprintf(stderr, "http: bind(%d) failed: %s\n", port_, strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    if (::listen(listen_fd_, backlog_) != 0) {
        fprintf(stderr, "http: listen() failed: %s\n", strerror(errno));
        ::close(listen_fd_);
        listen_fd_ = -1;
        return false;
    }
    fprintf(stderr, "http: listening on port %d\n", port_);
    return true;
}

class FdStreamWriter : public StreamWriter {
public:
    explicit FdStreamWriter(int fd) : fd_(fd) {}
    bool write_head(int status, const std::string & reason,
                    const std::map<std::string, std::string> & headers, bool streaming) override {
        streaming_ = streaming;
        std::string head = "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n";
        for (auto & kv : headers) head += kv.first + ": " + kv.second + "\r\n";
        if (!streaming_) head += "Content-Length: %LEN%\r\n";
        head += "Connection: close\r\n\r\n";
        head_ = std::move(head);
        if (streaming_) {
            if (!send_all(head_)) return false;
            head_sent_ = true;
        }
        return true;
    }
    bool write(const std::string & data) override {
        if (streaming_) return send_all(data);
        body_ += data;
        return true;
    }
    void finish() override {
        if (!streaming_ && !head_sent_) {
            std::string head = head_;
            std::string cl = std::to_string(body_.size());
            size_t pos = head.find("%LEN%");
            if (pos != std::string::npos) head.replace(pos, 5, cl);
            send_all(head);
            head_sent_ = true;
        }
        send_all(body_);
    }
    bool streaming_used() const override { return streaming_; }

private:
    bool send_all(const std::string & s) {
        size_t sent = 0;
        while (sent < s.size()) {
            ssize_t n = ::send(fd_, s.data() + sent, s.size() - sent, MSG_NOSIGNAL);
            if (n > 0) { sent += (size_t) n; continue; }
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        return true;
    }
    int fd_;
    bool streaming_ = false;
    bool head_sent_ = false;
    std::string head_;
    std::string body_;
};

bool HttpServer::read_request(int fd, HttpRequest & req) {
    // Read headers until CRLFCRLF (bounded).
    std::string buf;
    char chunk[4096];
    while (buf.find("\r\n\r\n") == std::string::npos) {
        ssize_t r = ::recv(fd, chunk, sizeof chunk, 0);
        if (r > 0) {
            buf.append(chunk, (size_t) r);
            if (buf.size() > 1 << 20) return false;  // header too large
        } else if (r == 0) {
            return false;
        } else if (errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }

    size_t head_end = buf.find("\r\n\r\n") + 4;
    std::string head = buf.substr(0, head_end);
    std::string rest = buf.substr(head_end);

    // Request line.
    size_t eol = head.find("\r\n");
    std::string line = head.substr(0, eol);
    size_t sp1 = line.find(' ');
    size_t sp2 = line.rfind(' ');
    if (sp1 == std::string::npos || sp2 == sp1) return false;
    req.method = line.substr(0, sp1);
    std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
    req.headers.clear();

    // Headers.
    size_t pos = eol + 2;
    while (pos < head_end) {
        size_t e = head.find("\r\n", pos);
        if (e == std::string::npos || e > head_end) break;
        std::string hl = head.substr(pos, e - pos);
        size_t colon = hl.find(':');
        if (colon != std::string::npos) {
            std::string k = to_lower(hl.substr(0, colon));
            std::string v = hl.substr(colon + 1);
            while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
            while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
            req.headers[k] = v;
        }
        pos = e + 2;
    }

    // Target: path + query.
    size_t q = target.find('?');
    if (q == std::string::npos) {
        req.path = target;
        req.query.clear();
    } else {
        req.path = target.substr(0, q);
        req.query = target.substr(q + 1);
    }

    auto conn = req.headers.find("connection");
    req.keep_alive = !(conn != req.headers.end() && to_lower(conn->second).find("close") != std::string::npos);

    // Body.
    auto cl = req.headers.find("content-length");
    size_t body_len = 0;
    if (cl != req.headers.end()) {
        body_len = (size_t) strtoull(cl->second.c_str(), nullptr, 10);
        if (body_len > 64 << 20) return false;  // body too large
    }
    req.body = rest;
    while (req.body.size() < body_len) {
        ssize_t r = ::recv(fd, chunk, sizeof chunk, 0);
        if (r > 0) req.body.append(chunk, (size_t) r);
        else if (r == 0) return false;
        else if (errno == EINTR) continue;
        else return false;
    }
    if (req.body.size() > body_len) req.body.resize(body_len);
    return true;
}

void HttpServer::handle_connection(int fd) {
    // Disable Nagle's algorithm: streaming (SSE) responses send many small
    // packets and Nagle + delayed ACK would stall each chunk for hundreds of
    // milliseconds.
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

    // Idle timeout for keep-alive connections.
    timeval tv{30, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);

    for (;;) {
        HttpRequest req;
        if (!read_request(fd, req)) break;
        if (req.path.empty() || req.path[0] != '/') {
            // Bad request; just close.
            break;
        }

        FdStreamWriter writer(fd);
        bool handled = false;
        for (auto & [m, p, h] : routes_) {
            if (m == req.method && p == req.path) {
                h(req, writer);
                handled = true;
                break;
            }
        }
        if (!handled) {
            std::map<std::string, std::string> hdrs{{"Content-Type", "application/json"}};
            writer.write_head(404, "Not Found", hdrs, false);
            writer.write("{\"error\":{\"message\":\"Not Found\",\"type\":\"not_found\",\"code\":404}}\n");
            writer.finish();
        }
        // Streaming (SSE) responses have no Content-Length and terminate with
        // [DONE]; close the connection so the client sees EOF immediately
        // instead of waiting for a keep-alive read timeout.
        if (writer.streaming_used()) break;
        if (!req.keep_alive) break;
    }
    ::close(fd);
}

void HttpServer::run() {
    for (;;) {
        sockaddr_in cli{};
        socklen_t clen = sizeof cli;
        int fd = ::accept(listen_fd_, (sockaddr *) &cli, &clen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "http: accept() failed: %s\n", strerror(errno));
            continue;
        }
        std::thread([this, fd]() { handle_connection(fd); }).detach();
    }
}
