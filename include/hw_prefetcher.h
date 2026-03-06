#ifndef HW_PREFETCHER_H
#define HW_PREFETCHER_H

// Hardware prefetcher control via MSR (Model-Specific Register)
// Requires root privileges and Secure Boot disabled in BIOS

// Save current hardware prefetcher state
bool save_prefetcher_state();

// Set hardware prefetcher state (enable=true for ON, false for OFF)
// Returns false if MSR access fails
bool set_prefetcher_state(bool enable);

// Restore saved hardware prefetcher state
void restore_prefetcher_state();

// Signal handler for cleanup (restores prefetcher state)
void signal_handler(int signum);

#endif // HW_PREFETCHER_H
