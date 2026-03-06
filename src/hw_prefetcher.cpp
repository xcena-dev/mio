#include "hw_prefetcher.h"

#include <signal.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "benchmark.h"

using namespace std;

// Global variables for hardware prefetcher control
static string saved_prefetcher_state;
static bool prefetcher_state_saved = false;

// Hardware Prefetcher Control Functions
bool save_prefetcher_state() {
  FILE* fp = popen("rdmsr 0x1A4 2>/dev/null", "r");
  if (!fp) {
    fprintf(stderr, "Warning: Failed to read MSR (rdmsr command failed)\n");
    return false;
  }

  char buffer[128];
  if (fgets(buffer, sizeof(buffer), fp)) {
    saved_prefetcher_state = buffer;
    // Remove newline
    saved_prefetcher_state.erase(
        saved_prefetcher_state.find_last_not_of(" \n\r\t") + 1);
    prefetcher_state_saved = true;
    pclose(fp);
    return true;
  }

  pclose(fp);
  return false;
}

bool set_prefetcher_state(bool enable) {
  const char* value = enable ? "0x0" : "0xf";
  char cmd[128];
  snprintf(cmd, sizeof(cmd), "wrmsr -a 0x1A4 %s 2>/dev/null", value);

  int ret = system(cmd);
  if (ret != 0) {
    fprintf(stderr,
            "Error: Failed to set prefetcher state (wrmsr command failed).\n");
    fprintf(stderr,
            "Please disable Secure Boot in BIOS to enable MSR access.\n");
    return false;
  }
  return true;
}

void restore_prefetcher_state() {
  if (!prefetcher_state_saved) {
    return;
  }

  char cmd[128];
  snprintf(cmd, sizeof(cmd), "wrmsr -a 0x1A4 0x%s 2>/dev/null",
           saved_prefetcher_state.c_str());
  int ret = system(cmd);
  (void)ret;  // Suppress unused warning
}

// Signal handler for cleanup
void signal_handler(int signum) {
  const char* signal_name = "UNKNOWN";
  switch (signum) {
    case SIGINT:
      signal_name = "SIGINT (Ctrl+C)";
      break;
    case SIGTERM:
      signal_name = "SIGTERM";
      break;
    case SIGSEGV:
      signal_name = "SIGSEGV (Segmentation fault)";
      break;
    case SIGBUS:
      signal_name = "SIGBUS (Bus error)";
      break;
  }
  fprintf(stderr, "\n========================================\n");
  fprintf(stderr, "FATAL: Received %s (signal %d)\n", signal_name, signum);
  if (signum == SIGSEGV || signum == SIGBUS) {
    fprintf(stderr, "This may be caused by:\n");
    fprintf(stderr, "  - DevDAX memory access out of bounds\n");
    fprintf(stderr, "  - DevDAX alignment issue (check: ndctl list -N)\n");
    fprintf(stderr, "  - Invalid memory mapping\n");
  }
  fprintf(stderr, "========================================\n");
  fprintf(stderr,
          "Cleaning up and exiting process (server will continue)...\n");
  restore_prefetcher_state();
  _exit(signum);
}