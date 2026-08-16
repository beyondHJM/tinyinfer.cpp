#pragma once
// Minimal HTTP/1.1 server for tinyinfer-server. No third-party dependencies:
// POSIX sockets, one thread per connection, keep-alive support, and a small
// StreamWriter abstraction that supports both buffered (Content-Length) and
// streaming (SSE) responses.

#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

struct HttpRequest {
    std::string method;
    std::string path;   // without query string
    std::string query;
    std::map<std::string, std::string> headers;  // lower-cased keys
    std::string body;
    bool keep_alive = true;
};

// Writer passed to handlers. Non-streaming handlers buffer the body and send it
// with Content-Length on finish(); streaming handlers (SSE) send the head and
// each chunk immediately and close the connection.
class StreamWriter {
public:
    virtual ~StreamWriter() = default;
    virtual bool write_head(int status, const std::string & reason,
                            const std::map<std::string, std::string> & headers,
                            bool streaming) = 0;
    virtual bool write(const std::string & data) = 0;
    virtual void finish() = 0;
    // True once write_head(..., streaming=true) was called. Streaming (SSE)
    // responses must close the connection when done (no Content-Length), so
    // the server should not keep the connection alive afterwards.
    virtual bool streaming_used() const = 0;
};

class HttpServer {
public:
    HttpServer(int port, int backlog = 64);
    ~HttpServer();
    bool start();  // bind + listen

    using Handler = std::function<void(const HttpRequest &, StreamWriter &)>;
    void handle(const std::string & method, const std::string & path, Handler h);

    void run();  // blocking accept loop

private:
    void handle_connection(int fd);
    bool read_request(int fd, HttpRequest & req);

    int port_;
    int backlog_;
    int listen_fd_ = -1;
    std::vector<std::tuple<std::string, std::string, Handler>> routes_;
};
