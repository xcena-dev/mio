#ifndef HW_PREFETCHER_H
#define HW_PREFETCHER_H

bool save_prefetcher_state();

bool set_prefetcher_state(bool enable);

void restore_prefetcher_state();

void signal_handler(int signum);

#endif
