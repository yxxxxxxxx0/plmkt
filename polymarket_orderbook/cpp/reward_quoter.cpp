// C++ port of reward_quoter.py's --all run loop (paper quoting; nothing is sent to the exchange).
// Market selection and the report stay in Python: `reward_quoter.py --all --select-only` writes the shard's
// state.json, this quotes it for --hours, writes the same fills.csv / q15.csv / state.json, and exits.
//
//   reward_quoter.exe <shard dir> [--shards 4] [--hours 6] [--latency 0.05] [--cacert path] [--selftest]
//
// Build: see build.sh. Deps: OpenSSL (TLS for the feed and REST), nlohmann/json (json.hpp, vendored).
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/ssl.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
using json = nlohmann::json;
using std::string;

// ---- constants, as in reward_quoter.py ------------------------------------------------------------------
static double LATENCY = 0.05;                     // s for an order or a cancel to take effect
const double PULL_MOVE = 0.05, PULL_S = 60;
const double VOL_MAX = 0.01, VOL_HALFLIFE = 10, TREND_TAU = 600, TREND_BANDS = 0.5, ADD_STEP = 0.01;
const double MAX_HOLD = 6 * 3600, MARKOUT_S = 600;
const int HIT_WINDOW = 6, HIT_MIN = 4;
const double HIT_RATE = 0.5, PAUSE_S = 86400;
const double ORDER_RATE = 40, ORDER_BURST = 60;   // Standard tier per signer; split over --shards
const double EPS = 1e-9;
const std::map<string, double> TAKER_FEE_RATE = {
    {"crypto", 0.07}, {"sports", 0.05}, {"economics", 0.05}, {"culture", 0.05}, {"weather", 0.05}, {"other", 0.05},
    {"finance", 0.04}, {"politics", 0.04}, {"mentions", 0.04}, {"tech", 0.04}, {"geopolitics", 0.00}};

static double now_s() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
static void log(const string& msg) {
    time_t t = time(nullptr);
    char b[32];
    strftime(b, sizeof b, "%Y-%m-%d %H:%M:%S", localtime(&t));
    printf("%s %s\n", b, msg.c_str());
    fflush(stdout);
}
static string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}
static double r6(double x) { return std::round(x * 1e6) / 1e6; }
static double tick_floor(double p, double tick) { return std::floor(r6(p / tick)) * tick; }
static double tick_ceil(double p, double tick) { return std::ceil(r6(p / tick)) * tick; }

// ---- book, orders, account (arb_scan.py's SimOrder / on_trade / on_book / Account) ----------------------
using Levels = std::map<double, double>;
struct Book { Levels bids, asks; };
static double level_at(const Levels& L, double p) {
    auto it = L.lower_bound(p - EPS);
    return it != L.end() && std::fabs(it->first - p) < EPS ? it->second : 0.0;
}

struct Order { string side; double price, size, ahead, live_at; };

static double on_trade(Order& o, double price, double size, double now) {
    if (now < o.live_at || o.size <= 0) return 0;
    bool through = o.side == "bid" ? price < o.price : price > o.price;
    double f;
    if (through) f = std::min(o.size, size);
    else if (std::fabs(price - o.price) < EPS) {
        f = std::min(o.size, std::max(0.0, size - o.ahead));
        o.ahead = std::max(0.0, o.ahead - size);
    } else return 0;
    o.size -= f;
    return f;
}
static double on_book(Order& o, double level_size, std::optional<double> best_opp, double now) {
    o.ahead = std::min(o.ahead, level_size);
    if (now >= o.live_at && best_opp && o.size > 0 &&
        ((o.side == "bid" && *best_opp <= o.price) || (o.side == "ask" && *best_opp >= o.price))) {
        double f = o.size;
        o.size = 0;
        return f;
    }
    return 0;
}

struct Account {
    double cash = 0;
    std::map<string, double> net;   // token -> net YES (+YES / -NO)
    void fill(const string& t, const string& side, double price, double qty) {
        double n = net[t];
        if (side == "bid") {
            double close = std::min(qty, std::max(0.0, -n));
            cash += close * (1 - price) - (qty - close) * price;
            net[t] = n + qty;
        } else {
            double close = std::min(qty, std::max(0.0, n));
            cash += close * price - (qty - close) * (1 - price);
            net[t] = n - qty;
        }
    }
    double cost(const string& t, const string& side, double price, double qty) const {
        auto it = net.find(t);
        double n = it == net.end() ? 0 : it->second;
        if (side == "bid") return std::max(0.0, qty - std::max(0.0, -n)) * price;
        return std::max(0.0, qty - std::max(0.0, n)) * (1 - price);
    }
};

static double score(const std::vector<std::pair<double, double>>& lv, double mid, double v, double minsz) {
    double s = 0;
    for (auto& [p, q] : lv)
        if (std::fabs(p - mid) < v && q >= minsz) s += std::pow((v - std::fabs(p - mid)) / v, 2) * q;
    return s;
}
static double score(const Levels& L, double mid, double v, double minsz) {
    return score(std::vector<std::pair<double, double>>(L.begin(), L.end()), mid, v, minsz);
}
static double q_min(double qb, double qa, double mid) {
    if (0.10 <= mid && mid <= 0.90) return std::max(std::min(qb, qa), std::max(qb, qa) / 3);
    return std::min(qb, qa);
}

// ---- TLS socket, HTTP GET, websocket client --------------------------------------------------------------
static SSL_CTX* CTX;
struct Tls {
    SOCKET s = INVALID_SOCKET;
    SSL* ssl = nullptr;
    bool open(const string& host) {
        addrinfo hints{}, *res;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), "443", &hints, &res)) return false;
        for (addrinfo* a = res; a; a = a->ai_next) {
            s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
            if (s != INVALID_SOCKET && connect(s, a->ai_addr, (int)a->ai_addrlen) == 0) break;
            if (s != INVALID_SOCKET) closesocket(s);
            s = INVALID_SOCKET;
        }
        freeaddrinfo(res);
        if (s == INVALID_SOCKET) return false;
        ssl = SSL_new(CTX);
        SSL_set_fd(ssl, (int)s);
        SSL_set_tlsext_host_name(ssl, host.c_str());
        SSL_set1_host(ssl, host.c_str());          // certificate must match the host
        return SSL_connect(ssl) == 1;
    }
    bool write(const string& d) { return SSL_write(ssl, d.data(), (int)d.size()) == (int)d.size(); }
    bool readable(int ms) {                       // data buffered in OpenSSL or on the socket
        if (SSL_pending(ssl) > 0) return true;
        fd_set f;
        FD_ZERO(&f);
        FD_SET(s, &f);
        timeval tv{ms / 1000, (ms % 1000) * 1000};
        return select(0, &f, nullptr, nullptr, &tv) > 0;
    }
    int read(char* b, int n) { return SSL_read(ssl, b, n); }
    void close() {
        if (ssl) SSL_free(ssl);
        if (s != INVALID_SOCKET) closesocket(s);
        ssl = nullptr;
        s = INVALID_SOCKET;
    }
};

static string url_encode(const string& v) {
    string o;
    for (unsigned char c : v)
        o += isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~' ? string(1, c) : fmt("%%%02X", c);
    return o;
}

// GET https://host/path?query -> parsed JSON (null on any failure, like reward_quoter.get after its retries).
static json http_get(const string& host, const string& path) {
    for (int i = 0; i < 4; i++) {
        Tls t;
        if (t.open(host) && t.write("GET " + path + " HTTP/1.1\r\nHost: " + host +
                                    "\r\nAccept: application/json\r\nConnection: close\r\n\r\n")) {
            string r;
            char b[16384];
            int n;
            while ((n = t.read(b, sizeof b)) > 0) r.append(b, n);
            t.close();
            size_t h = r.find("\r\n\r\n");
            if (h != string::npos && r.compare(0, 12, "HTTP/1.1 200") == 0) {
                string head = r.substr(0, h), body = r.substr(h + 4);
                std::transform(head.begin(), head.end(), head.begin(), ::tolower);
                if (head.find("transfer-encoding: chunked") != string::npos) {
                    string out;
                    size_t p = 0;
                    while (p < body.size()) {
                        size_t e = body.find("\r\n", p);
                        if (e == string::npos) break;
                        size_t len = std::stoul(body.substr(p, e - p), nullptr, 16);
                        if (!len) break;
                        out += body.substr(e + 2, len);
                        p = e + 2 + len + 2;
                    }
                    body = out;
                }
                try { return json::parse(body); } catch (...) {}
            }
        }
        t.close();
        std::this_thread::sleep_for(std::chrono::seconds(2 + 3 * i));
    }
    return nullptr;
}

struct Ws {
    Tls t;
    string buf, msg;
    bool open(const string& host, const string& path) {
        if (!t.open(host)) return false;
        unsigned char k[16];
        RAND_bytes(k, 16);
        unsigned char key[32];
        EVP_EncodeBlock(key, k, 16);
        if (!t.write("GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                     "Sec-WebSocket-Key: " + string((char*)key) + "\r\nSec-WebSocket-Version: 13\r\n\r\n"))
            return false;
        while (buf.find("\r\n\r\n") == string::npos) {
            char b[4096];
            int n = t.read(b, sizeof b);
            if (n <= 0) return false;
            buf.append(b, n);
        }
        if (buf.compare(0, 12, "HTTP/1.1 101") != 0) return false;
        buf.erase(0, buf.find("\r\n\r\n") + 4);
        return true;
    }
    bool send(int op, const string& d) {          // client frames are masked
        string f(1, (char)(0x80 | op));
        size_t n = d.size();
        if (n < 126) f += (char)(0x80 | n);
        else if (n < 65536) { f += (char)(0x80 | 126); f += (char)(n >> 8); f += (char)n; }
        else { f += (char)(0x80 | 127); for (int i = 7; i >= 0; i--) f += (char)(n >> (8 * i)); }
        unsigned char m[4];
        RAND_bytes(m, 4);
        f.append((char*)m, 4);
        for (size_t i = 0; i < n; i++) f += (char)(d[i] ^ m[i % 4]);
        return t.write(f);
    }
    // One complete text message into out. 1 = message, 0 = nothing within ms, -1 = connection gone.
    int next(string& out, int ms, double& last_rx) {
        for (;;) {
            if (buf.size() >= 2) {
                unsigned char b0 = buf[0], b1 = buf[1];
                size_t len = b1 & 0x7f, hdr = 2;
                if (len == 126) { if (buf.size() < 4) goto more; len = (unsigned char)buf[2] << 8 | (unsigned char)buf[3]; hdr = 4; }
                else if (len == 127) {
                    if (buf.size() < 10) goto more;
                    len = 0;
                    for (int i = 0; i < 8; i++) len = len << 8 | (unsigned char)buf[2 + i];
                    hdr = 10;
                }
                if (buf.size() < hdr + len) goto more;
                string pay = buf.substr(hdr, len);
                buf.erase(0, hdr + len);
                int op = b0 & 0x0f;
                if (op == 0x8) return -1;
                if (op == 0x9) { send(0xA, pay); continue; }
                if (op == 0xA) continue;
                msg += pay;                        // text (0x1) or continuation (0x0)
                if (b0 & 0x80) { out.swap(msg); msg.clear(); return 1; }
                continue;
            }
        more:
            if (!t.readable(ms)) return 0;
            char b[65536];
            int n = t.read(b, sizeof b);
            if (n <= 0) return -1;
            buf.append(b, n);
            last_rx = now_s();
        }
    }
};

// ---- the quoter ------------------------------------------------------------------------------------------
struct Market { string cond, q, yes, fee; double rate, v, minsz, tick, size; };
struct Entry { string side; double fair, price, opened; };
struct Tok {                                       // per YES token
    Market m;
    Book b;
    std::map<string, Order> orders;                 // side -> order
    std::vector<std::pair<double, Order>> dying;    // (cancel lands at, order): still fillable until then
    std::deque<std::pair<double, double>> hist;     // (time, mid) for the pull rule
    double pulled = 0;
    bool blind = false, done = false;
    double var = 0; int vn = 0;
    std::optional<std::pair<double, double>> last, slow;
    int a_mins = 0, a_quoted = 0; double a_reward = 0;
};

struct Quoter {
    string dir;
    json st;
    std::map<string, Tok> T;
    Account acct;
    double reward = 0;
    int n_fill = 0, n_limited = 0;
    std::map<string, Entry> entry;
    std::map<string, std::vector<int>> outcomes;
    std::map<string, std::pair<double, string>> paused;
    std::vector<std::tuple<double, string, string, double>> pending;   // (due, token, side, price)
    double rate, burst, tokens, t_tok;
    FILE *fw = nullptr, *qw = nullptr;
    std::mutex mu;

    Quoter(const string& d, int shards) : dir(d) {
        st = json::parse(std::ifstream(dir + "/state.json"));
        for (auto& m : st["markets"]) {
            Tok k;
            k.m = {m["cond"], m["q"], m["yes"], m.value("fee", ""), m["rate"].get<double>(), m["v"].get<double>(),
                   m["minsz"].get<double>(), m["tick"].get<double>(), m["size"].get<double>()};
            T[k.m.yes] = k;
        }
        acct.cash = st["cash"];
        for (auto& [t, n] : st["net"].items()) acct.net[t] = n;
        reward = st["reward"];
        for (auto k : {"entry", "outcomes", "paused"})   // iterate st's own objects: value() would return a temporary
            if (!st.contains(k) || !st[k].is_object()) st[k] = json::object();
        for (auto& [t, e] : st["entry"].items()) entry[t] = {e["side"], e["fair"], e["price"], e["opened"]};
        for (auto& [t, h] : st["outcomes"].items()) outcomes[t] = h.get<std::vector<int>>();
        for (auto& [t, p] : st["paused"].items()) paused[t] = {p[0], p[1]};
        rate = ORDER_RATE / std::max(1, shards);
        burst = ORDER_BURST / std::max(1, shards);
        tokens = burst;
        t_tok = now_s();
        fw = csv(dir + "/fills.csv", "ts,cond,side,price,qty,mid,net_after,how");
        qw = csv(dir + "/q15.csv", "ts,cond,mid,best_bid,best_ask,mins,quoted_mins,reward,net,cash");
    }
    static FILE* csv(const string& p, const char* header) {
        FILE* f = fopen(p.c_str(), "ab");
        fseek(f, 0, SEEK_END);                         // append mode reads 0 until the first write
        if (ftell(f) == 0) fprintf(f, "%s\n", header);
        return f;
    }
    double net(const string& t) const { auto it = acct.net.find(t); return it == acct.net.end() ? 0 : it->second; }

    std::optional<double> mid(const string& t) {
        auto& b = T[t].b;
        if (b.bids.empty() || b.asks.empty()) return std::nullopt;
        return (b.bids.rbegin()->first + b.asks.begin()->first) / 2;
    }

    bool take_order_token(double now) {
        tokens = std::min(burst, tokens + (now - t_tok) * rate);
        t_tok = now;
        if (tokens < 1) { n_limited++; return false; }
        tokens -= 1;
        return true;
    }
    void drop(const string& t, const string& side = "") {   // cancel: fillable until it lands
        auto& k = T[t];
        double now = now_s();
        for (string s : side.empty() ? std::vector<string>{"bid", "ask"} : std::vector<string>{side}) {
            auto it = k.orders.find(s);
            if (it == k.orders.end()) continue;
            if (it->second.size > 0) k.dying.push_back({now + LATENCY, it->second});
            k.orders.erase(it);
        }
    }
    std::vector<Order*> resting(const string& t, double now) {
        auto& k = T[t];
        std::erase_if(k.dying, [&](auto& d) { return d.first <= now || d.second.size <= 0; });
        std::vector<Order*> r;
        for (auto& [s, o] : k.orders) r.push_back(&o);
        for (auto& [x, o] : k.dying) r.push_back(&o);
        return r;
    }

    void fill(const string& t, const string& side, double price, double qty, const char* how) {
        if (qty <= 0) return;
        double before = net(t);
        acct.fill(t, side, price, qty);
        n_fill++;
        double now = now_s(), after = net(t);
        auto m = mid(t);
        pending.push_back({now + MARKOUT_S, t, side, price});
        if (std::fabs(after) < 1) entry.erase(t);
        else if (std::fabs(after) > std::fabs(before) || before * after < 0) {   // opened or flipped
            auto e = entry.find(t);
            if (e == entry.end() || e->second.side != side || before * after < 0)
                entry[t] = {side, m ? *m : price, price, now};
            else e->second.price = price;
        }
        fprintf(fw, "%.3f,%s,%s,%.10g,%.4f,%s,%.4f,%s\n", now, T[t].m.cond.c_str(), side.c_str(), price, qty,
                m ? fmt("%.10g", *m).c_str() : "", after, how);
        fflush(fw);
    }

    void on_trade_(const string& t, double price, double size) {
        double now = now_s();
        for (Order* o : resting(t, now)) {
            string side = o->side;
            double p = o->price, f = on_trade(*o, price, size, now);
            fill(t, side, p, f, "print");
        }
    }
    void on_book_(const string& t) {
        auto& b = T[t].b;
        double now = now_s();
        for (Order* o : resting(t, now)) {
            double lvl = level_at(o->side == "bid" ? b.bids : b.asks, o->price);
            std::optional<double> opp;
            if (o->side == "bid" && !b.asks.empty()) opp = b.asks.begin()->first;
            if (o->side == "ask" && !b.bids.empty()) opp = b.bids.rbegin()->first;
            string side = o->side;
            double p = o->price, f = on_book(*o, lvl, opp, now);
            fill(t, side, p, f, "cross");
        }
    }

    const char* no_open(const string& t, double mid, double now, string& why) {
        auto& k = T[t];
        auto p = paused.find(t);
        if (p != paused.end() && p->second.first > now) return (why = p->second.second).c_str();
        if (!k.last) k.last = {{now, mid}};
        else if (now - k.last->first >= 60) {
            double a = 1 - std::pow(0.5, 1 / VOL_HALFLIFE), d2 = std::pow(mid - k.last->second, 2);
            k.var = k.vn == 0 ? d2 : (1 - a) * k.var + a * d2;
            k.vn++;
            k.last = {{now, mid}};
        }
        double slow = !k.slow ? mid : k.slow->second + (1 - std::exp(-(now - k.slow->first) / TREND_TAU)) * (mid - k.slow->second);
        k.slow = {{now, slow}};
        if (k.vn < 3) return (why = "warming up").c_str();
        if (std::sqrt(k.var) > VOL_MAX) return (why = fmt("volatile (%.1fc/min)", std::sqrt(k.var) * 100)).c_str();
        if (std::fabs(mid - slow) > TREND_BANDS * k.m.v) return (why = fmt("trending (%+.1fc from slow fair)", (mid - slow) * 100)).c_str();
        return nullptr;
    }

    double free_cash(const string& t, const string& side) {
        double used = 0;
        for (auto& [u, k] : T) {
            for (auto& [s, o] : k.orders)
                if (!(u == t && s == side)) used += acct.cost(u, s, o.price, o.size);
            for (auto& [x, o] : k.dying) used += acct.cost(u, o.side, o.price, o.size);
        }
        return acct.cash - used;
    }

    void time_stop(const string& t, double n) {   // taker exit at the top level, fee folded into the price
        auto& k = T[t];
        string f = k.m.fee;
        if (auto i = f.find("_fees"); i != string::npos) f.erase(i, 5);
        f = f.rfind("sports", 0) == 0 ? "sports" : f.substr(0, f.find("_v"));
        double rate = (f.empty() || f == "zero") ? 0 : (TAKER_FEE_RATE.count(f) ? TAKER_FEE_RATE.at(f) : 0.05);
        if (n > 0 && !k.b.bids.empty()) {
            auto [p, q] = *k.b.bids.rbegin();
            fill(t, "ask", p - rate * p * (1 - p), std::min(n, q), "time_stop");
        } else if (n < 0 && !k.b.asks.empty()) {
            auto [p, q] = *k.b.asks.begin();
            fill(t, "bid", p + rate * p * (1 - p), std::min(-n, q), "time_stop");
        }
        drop(t);
        if (entry.count(t)) entry[t].opened = now_s() - MAX_HOLD + 60;   // the rest: retry in 1 min
    }

    void requote(const string& t, double now) {
        auto& k = T[t];
        auto& m = k.m;
        auto& b = k.b;
        if (k.blind || k.done) return;
        auto md = mid(t);
        if (!md) { drop(t); return; }
        double mid = *md;
        k.hist.push_back({now, mid});
        while (!k.hist.empty() && k.hist.front().first < now - 30) k.hist.pop_front();
        double hi = -1, lo = 2;
        for (auto& [x, y] : k.hist) hi = std::max(hi, y), lo = std::min(lo, y);
        if (hi - lo >= PULL_MOVE) k.pulled = now + PULL_S;
        if (now < k.pulled || b.asks.begin()->first - b.bids.rbegin()->first > 2 * m.v) { drop(t); return; }
        double n = net(t), sz = m.size;
        auto e = entry.find(t);
        bool has_e = e != entry.end();
        if (has_e && now - e->second.opened >= MAX_HOLD) { time_stop(t, n); return; }
        string why_s;
        bool why = no_open(t, mid, now, why_s) != nullptr;
        double c = mid - m.v / 2 * n / (2 * sz);       // inventory skew against the position
        std::map<string, double> tgt = {{"bid", tick_floor(c - m.v / 2, m.tick)}, {"ask", tick_ceil(c + m.v / 2, m.tick)}};
        if (has_e) {                                   // frozen exit: never close worse than the fair at entry
            if (n > 0) tgt["ask"] = std::max(tgt["ask"], tick_ceil(e->second.fair, m.tick));
            else tgt["bid"] = std::min(tgt["bid"], tick_floor(e->second.fair, m.tick));
        }
        tgt["bid"] = std::min(tgt["bid"], r6(b.asks.begin()->first - m.tick));     // never cross
        tgt["ask"] = std::max(tgt["ask"], r6(b.bids.rbegin()->first + m.tick));
        for (string side : {"bid", "ask"}) {
            double p = tgt[side];
            double closing = side == "bid" ? std::max(0.0, -n) : std::max(0.0, n);
            double room = side == "bid" ? 2 * sz - n : 2 * sz + n;
            bool stacked = has_e && e->second.side == side &&
                           (side == "bid" ? p > e->second.price - ADD_STEP : p < e->second.price + ADD_STEP);
            bool opening = !(why || stacked);
            double size = opening ? std::min(sz, room) : std::min(sz, closing);
            if (size < (opening ? m.minsz : 5) || !(0 < p && p < 1)) { drop(t, side); continue; }
            auto o = k.orders.find(side);
            if (o != k.orders.end() && std::fabs(o->second.price - p) < 1e-9 && 0 < o->second.size && o->second.size <= size)
                continue;
            if (acct.cost(t, side, p, size) > free_cash(t, side)) { drop(t, side); continue; }
            if (!take_order_token(now)) continue;      // rate limited: the old quote stays
            drop(t, side);                             // a replace is a cancel plus a new order
            double ahead = level_at(side == "bid" ? b.bids : b.asks, p);
            k.orders[side] = {side, p, size, ahead, now + LATENCY};
        }
    }

    void score_markouts() {
        double now = now_s();
        std::vector<std::tuple<double, string, string, double>> keep;
        for (auto& x : pending) {
            auto& [due, t, side, price] = x;
            if (due > now) { keep.push_back(x); continue; }
            auto md = mid(t);
            if (!md) continue;
            bool won = side == "bid" ? *md > price : *md < price;
            auto& h = outcomes[t];
            h.push_back(won);
            if ((int)h.size() > HIT_WINDOW) h.erase(h.begin(), h.end() - HIT_WINDOW);
            int s = 0;
            for (int v : h) s += v;
            if ((int)h.size() >= HIT_MIN && (double)s / h.size() < HIT_RATE) {
                paused[t] = {now + PAUSE_S, fmt("fills won %d/%zu", s, h.size())};
                log(fmt("paused 24h: %.50s (fills won %d/%zu at +%d min)", T[t].m.q.c_str(), s, h.size(), (int)MARKOUT_S / 60));
                outcomes.erase(t);
            }
        }
        pending.swap(keep);
    }

    void sample() {
        double now = now_s();
        bool row = (long long)(now / 60) % 15 == 0;
        for (auto& [t, k] : T) {
            auto md = mid(t);
            if (!md) continue;
            double mid = *md;
            auto& m = k.m;
            double others = q_min(score(k.b.bids, mid, m.v, m.minsz), score(k.b.asks, mid, m.v, m.minsz), mid);
            double qs[2] = {0, 0};
            int i = 0;
            for (string s : {"bid", "ask"}) {
                auto o = k.orders.find(s);
                if (o != k.orders.end() && now >= o->second.live_at && o->second.size > 0)
                    qs[i] = score(Levels{{o->second.price, o->second.size}}, mid, m.v, m.minsz);
                i++;
            }
            double mine = q_min(qs[0], qs[1], mid);
            double share = mine > 0 ? mine / (mine + others) : 0, r = m.rate / 1440 * share;
            reward += r;
            k.a_mins++; k.a_quoted += mine > 0; k.a_reward += r;
            if (row) {
                fprintf(qw, "%.0f,%s,%.10g,%.10g,%.10g,%d,%d,%.5f,%.4f,%.2f\n", std::round(now), m.cond.c_str(), mid,
                        k.b.bids.rbegin()->first, k.b.asks.begin()->first, k.a_mins, k.a_quoted, k.a_reward, net(t), acct.cash);
                k.a_mins = k.a_quoted = 0; k.a_reward = 0;
            }
        }
        fflush(qw);
        score_markouts();
        save();
    }
    void save() {
        st["cash"] = acct.cash;
        st["net"] = acct.net;
        st["reward"] = reward;
        json e = json::object(), o = json::object(), p = json::object();
        for (auto& [t, x] : entry) e[t] = {{"side", x.side}, {"fair", x.fair}, {"price", x.price}, {"opened", x.opened}};
        for (auto& [t, h] : outcomes) o[t] = h;
        for (auto& [t, x] : paused) p[t] = {x.first, x.second};
        st["entry"] = e; st["outcomes"] = o; st["paused"] = p;
        string path = dir + "/state.json", tmp = path + ".tmp";
        std::ofstream(tmp) << st.dump(1);
        if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))   // e.g. antivirus holding it
            log(fmt("state save failed (%lu), retrying next minute", GetLastError()));
    }

    static void apply_levels(Levels& L, const json& arr) {
        L.clear();
        for (auto& l : arr) {
            double s = std::stod(l["size"].get<string>());
            if (s > 0) L[std::stod(l["price"].get<string>())] = s;
        }
    }
    void handle(const json& msg) {
        string et = msg.value("event_type", ""), t = msg.value("asset_id", "");
        double now = now_s();
        if (et == "book" && T.count(t)) {
            apply_levels(T[t].b.bids, msg.value("bids", json::array()));
            apply_levels(T[t].b.asks, msg.value("asks", json::array()));
            on_book_(t);                               // anything the market moved through while blind fills
            T[t].blind = false;
            requote(t, now);
        } else if (et == "price_change") {
            std::set<string> seen;
            json ch = msg.contains("price_changes") ? msg["price_changes"] : msg.value("changes", json::array());
            for (auto& c : ch) {
                string u = c.value("asset_id", t);
                if (!T.count(u)) continue;
                double p = std::stod(c["price"].get<string>()), s = std::stod(c["size"].get<string>());
                string sd = c["side"];
                Levels& L = sd == "BUY" || sd == "BID" ? T[u].b.bids : T[u].b.asks;
                if (s <= 0) L.erase(p); else L[p] = s;
                on_book_(u);
                seen.insert(u);
            }
            for (auto& u : seen) requote(u, now);
        } else if (et == "last_trade_price" && T.count(t)) {   // YES-token prints only, as in the Python
            json sz = msg.value("size", json("0"));
            on_trade_(t, std::stod(msg["price"].get<string>()), sz.is_string() ? std::stod(sz.get<string>()) : sz.get<double>());
        } else if (et == "tick_size_change" && T.count(t)) {
            T[t].m.tick = std::stod(msg["new_tick_size"].get<string>());
        }
    }

    void feed(std::vector<string> toks) {
        for (;;) {
            Ws ws;
            json sub = {{"assets_ids", toks}, {"type", "market"}};
            if (ws.open("ws-subscriptions-clob.polymarket.com", "/ws/market") && ws.send(0x1, sub.dump())) {
                double last_rx = now_s(), last_ping = now_s();
                string raw;
                for (;;) {
                    int r = ws.next(raw, 1000, last_rx);
                    if (r < 0 || now_s() - last_rx > 20) break;         // gone, or silent past the ping timeout
                    if (now_s() - last_ping >= 10) { ws.send(0x9, ""); last_ping = now_s(); }
                    if (r == 0) continue;
                    json P;
                    try { P = json::parse(raw); } catch (...) { continue; }
                    std::lock_guard<std::mutex> g(mu);
                    try {
                        if (P.is_array()) { for (auto& m : P) if (m.is_object()) handle(m); }
                        else if (P.is_object()) handle(P);
                    } catch (const std::exception& e) { log(fmt("bad message (%s)", e.what())); }
                }
            }
            ws.t.close();
            {
                std::lock_guard<std::mutex> g(mu);
                for (auto& t : toks) T[t].blind = true;                // orders stay live at the exchange, unseen
            }
            log("feed reconnect");
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
    }

    void refresh_rates() {                          // one paged listing; HTTP outside the lock
        std::map<string, double> R;
        string cur;
        while (cur != "LTE=") {
            json d = http_get("clob.polymarket.com", "/rewards/markets/current?next_cursor=" + url_encode(cur));
            if (d.is_null()) return;                    // keep the old rates rather than zero them
            if (d.contains("data") && d["data"].is_array())
                for (auto& r : d["data"])
                    if (r["condition_id"].is_string() && r["total_daily_rate"].is_number())
                        R[r["condition_id"]] = r["total_daily_rate"].get<double>();
            cur = d.contains("next_cursor") && d["next_cursor"].is_string() ? d["next_cursor"].get<string>() : "";
            if (cur.empty()) cur = "LTE=";                 // the API ends the listing with null or "LTE="
        }
        std::lock_guard<std::mutex> g(mu);
        for (auto& [t, k] : T) k.m.rate = R.count(k.m.cond) ? R[k.m.cond] : 0;
    }
    void settle_resolved() {
        std::vector<string> held;
        {
            std::lock_guard<std::mutex> g(mu);
            for (auto& [t, k] : T) if (std::fabs(net(t)) >= 1 && !k.done) held.push_back(k.m.cond);
        }
        std::map<string, int> out;                      // YES token -> payout
        for (size_t i = 0; i < held.size(); i += 50) {
            string q = "/markets?closed=true&limit=50";
            for (size_t j = i; j < std::min(held.size(), i + 50); j++) q += "&condition_ids=" + held[j];
            json d = http_get("gamma-api.polymarket.com", q);
            if (!d.is_array()) continue;
            for (auto& g : d) {
                try {
                    double px = std::stod(json::parse(g["outcomePrices"].get<string>())[0].get<string>());
                    string yes = json::parse(g["clobTokenIds"].get<string>())[0];
                    if (g.value("closed", false) && (px >= 0.99 || px <= 0.01)) out[yes] = (int)std::lround(px);
                } catch (...) {}
            }
        }
        std::lock_guard<std::mutex> g(mu);
        for (auto& [t, px] : out) {
            if (!T.count(t)) continue;
            double n = net(t);
            drop(t);
            T[t].dying.clear();
            if (n > 0) fill(t, "ask", px, n, "settle");
            else if (n < 0) fill(t, "bid", px, -n, "settle");
            T[t].done = true;
            log(fmt("settled %.50s at %d: %+.0f YES", T[t].m.q.c_str(), px, n));
        }
    }

    void run(double hours) {
        std::vector<string> toks;
        for (auto& [t, k] : T) toks.push_back(t);
        for (size_t i = 0; i < toks.size(); i += 200)
            std::thread(&Quoter::feed, this, std::vector<string>(toks.begin() + i, toks.begin() + std::min(toks.size(), i + 200))).detach();
        double end = now_s() + hours * 3600, next_sweep = now_s() + 30;
        std::mt19937 rng(std::random_device{}());
        auto next_sample = [&] { double n = now_s(); return n - std::fmod(n, 60) + 60 + std::uniform_real_distribution<>(0, 59)(rng); };
        double t_sample = next_sample();
        refresh_rates();
        long long last_refresh = (long long)(now_s() / 60);
        while (now_s() < end) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            double now = now_s();
            long long minute = (long long)(now / 60);
            if (minute % 120 == 0 && minute != last_refresh) {
                last_refresh = minute;
                refresh_rates();
                settle_resolved();
            }
            std::lock_guard<std::mutex> g(mu);
            if (now >= next_sweep) {                    // slow sweep: time stop and quiet books
                for (auto& [t, k] : T) requote(t, now);
                next_sweep = now + 30;
            }
            if (now >= t_sample) {
                sample();
                t_sample = next_sample();
                if (minute % 15 == 0) {
                    string pos;
                    for (auto& [t, n] : acct.net)
                        if (std::fabs(n) > 1e-6 && T.count(t)) pos += fmt("%s%.20s: %+.0f", pos.empty() ? "" : ", ", T[t].m.q.c_str(), n);
                    log(fmt("reward $%.2f  fills %d  cash $%.2f  rate-limited %d  positions {%s}", reward, n_fill, acct.cash, n_limited, pos.c_str()));
                }
            }
        }
        std::lock_guard<std::mutex> g(mu);
        save();
        log("reselecting");
    }
};

// ---- self-check: the same cases as test_arb_scan.py's queue/account checks plus the realism rules ----------
static void selftest() {
    Order o{"bid", 0.30, 100, 150, 0};
    assert(on_trade(o, 0.30, 100, 1) == 0 && o.ahead == 50);
    assert(on_trade(o, 0.30, 80, 1) == 30 && o.size == 70);
    assert(on_trade(o, 0.29, 20, 1) == 20);
    assert(on_book(o, 0.0, 0.30, 1) == 50 && o.size == 0);
    Order late{"ask", 0.40, 10, 0, 5};
    assert(on_trade(late, 0.41, 10, 1) == 0);
    Account acc;
    acc.cash = 50;
    acc.fill("t", "bid", 0.30, 100);
    acc.fill("t", "ask", 0.32, 150);
    assert(std::fabs(acc.cash - 18.0) < 1e-9 && acc.net["t"] == -50);
    assert(std::fabs(acc.cost("t", "bid", 0.3, 80) - 30 * 0.3) < 1e-12);
    assert(std::fabs(score(Levels{{0.49, 100}}, 0.50, 0.03, 20) - 100 * std::pow(2.0 / 3, 2)) < 1e-9);
    assert(score(Levels{{0.49, 10}}, 0.50, 0.03, 20) == 0);
    assert(q_min(9, 0, 0.5) == 3 && q_min(9, 0, 0.95) == 0);
    assert(std::fabs(tick_floor(0.4875, 0.01) - 0.48) < 1e-12 && std::fabs(tick_ceil(0.5125, 0.01) - 0.52) < 1e-12);
    puts("selftest ok");
}

int main(int argc, char** argv) {
    string dir, cacert = "D:/plmkt/.venv/Lib/site-packages/certifi/cacert.pem";
    int shards = 4;
    double hours = 6;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--selftest") { selftest(); return 0; }
        else if (a == "--shards") shards = std::stoi(argv[++i]);
        else if (a == "--hours") hours = std::stod(argv[++i]);
        else if (a == "--latency") LATENCY = std::stod(argv[++i]);
        else if (a == "--cacert") cacert = argv[++i];
        else dir = a;
    }
    if (dir.empty()) { fprintf(stderr, "usage: reward_quoter <shard dir> [--shards N] [--hours H] [--latency S]\n"); return 2; }
    WSADATA w;
    WSAStartup(MAKEWORD(2, 2), &w);
    CTX = SSL_CTX_new(TLS_client_method());
    SSL_CTX_set_verify(CTX, SSL_VERIFY_PEER, nullptr);
    if (SSL_CTX_load_verify_locations(CTX, cacert.c_str(), nullptr) != 1) { log("cannot load CA bundle " + cacert); return 1; }
    Quoter q(dir, shards);
    log(fmt("surveying %zu markets, cash $%.2f, reward so far $%.2f", q.T.size(), q.acct.cash, q.reward));
    q.run(hours);
    fflush(nullptr);
    _exit(0);                                         // feed threads are detached and blocking
}
