// progress.h -- a single-line terminal progress meter for the search.
//
// Header-only, no CUDA, no dependency on the rest of the program, so it can be
// unit-tested on its own. render() returns the string it would print, which is
// what the tests check; print() adds the carriage return and flush.
//
//   [######################-------]  73.4%  1.18 Gkey/s  2m18s / eta 51s  found 0
//
// The rate shown is a rolling average over a short window, so the ETA tracks
// current speed rather than being dragged by a slow start.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <deque>
#include <chrono>

class Progress {
public:
  // total: total keys in the whole run. width: inner bar cells.
  // isatty: if false, render() still works but print() emits newlines instead
  //         of carriage returns, so logs stay readable when redirected.
  Progress(uint64_t total, int width = 30, bool isatty = true)
      : total_(total ? total : 1), width_(width), tty_(isatty) {}

  // Feed cumulative keys done and seconds elapsed since the run started.
  // Returns the meter string (no trailing newline / CR).
  std::string render(uint64_t done, double elapsed, int found) {
    if (done > total_) done = total_;
    push(done, elapsed);

    double frac = (double)done / (double)total_;
    double rate = rolling_rate();               // keys/sec, recent window
    double remaining = (rate > 1e-9) ? (double)(total_ - done) / rate : -1.0;

    int fill = (int)(frac * width_ + 0.5);
    if (fill > width_) fill = width_;

    std::string bar;
    bar.reserve(width_ + 2);
    bar += '[';
    for (int i = 0; i < width_; i++) bar += (i < fill) ? '#' : '-';
    bar += ']';

    char buf[256];
    snprintf(buf, sizeof(buf), "%s %5.1f%%  %s  %s / eta %s  found %d",
             bar.c_str(), frac * 100.0,
             rate_str(rate).c_str(),
             dur_str(elapsed).c_str(),
             remaining < 0 ? std::string("--").c_str() : dur_str(remaining).c_str(),
             found);
    last_len_ = (int)strlen(buf);
    return std::string(buf);
  }

  void print(uint64_t done, double elapsed, int found) {
    std::string s = render(done, elapsed, found);
    if (tty_) {
      // pad to overwrite any longer previous line, then carriage-return
      std::string pad;
      if ((int)s.size() < prev_print_len_) pad.assign(prev_print_len_ - s.size(), ' ');
      printf("\r%s%s", s.c_str(), pad.c_str());
      prev_print_len_ = (int)s.size();
    } else {
      printf("%s\n", s.c_str());
    }
    fflush(stdout);
  }

  // Call once when the run finishes to move off the meter line.
  void finish() {
    if (tty_) printf("\n");
    fflush(stdout);
  }

  // ---- formatting helpers, exposed for testing ----
  static std::string rate_str(double keys_per_sec) {
    const char *u[] = {"key/s", "Kkey/s", "Mkey/s", "Gkey/s", "Tkey/s"};
    double v = keys_per_sec;
    int i = 0;
    while (v >= 1000.0 && i < 4) { v /= 1000.0; i++; }
    char b[48];
    snprintf(b, sizeof(b), "%6.2f %s", v, u[i]);
    return b;
  }

  static std::string dur_str(double seconds) {
    if (seconds < 0) seconds = 0;
    uint64_t s = (uint64_t)(seconds + 0.5);
    uint64_t d = s / 86400; s %= 86400;
    uint64_t h = s / 3600;  s %= 3600;
    uint64_t m = s / 60;    s %= 60;
    char b[48];
    if (d) snprintf(b, sizeof(b), "%llud%lluh%llum", (unsigned long long)d,
                    (unsigned long long)h, (unsigned long long)m);
    else if (h) snprintf(b, sizeof(b), "%lluh%llum%llus", (unsigned long long)h,
                         (unsigned long long)m, (unsigned long long)s);
    else if (m) snprintf(b, sizeof(b), "%llum%llus", (unsigned long long)m,
                         (unsigned long long)s);
    else snprintf(b, sizeof(b), "%llus", (unsigned long long)s);
    return b;
  }

private:
  struct Sample { double t; uint64_t done; };
  void push(uint64_t done, double elapsed) {
    win_.push_back({elapsed, done});
    // keep roughly the last 10 seconds, and at least 2 samples
    while (win_.size() > 2 && (elapsed - win_.front().t) > 10.0) win_.pop_front();
  }
  double rolling_rate() const {
    if (win_.size() < 2) return 0.0;
    const Sample &a = win_.front();
    const Sample &b = win_.back();
    double dt = b.t - a.t;
    if (dt <= 1e-9) return 0.0;
    return (double)(b.done - a.done) / dt;
  }

  uint64_t total_;
  int width_;
  bool tty_;
  int last_len_ = 0;
  int prev_print_len_ = 0;
  std::deque<Sample> win_;
};
