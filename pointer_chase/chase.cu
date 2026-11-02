// Your kernel. Write it yourself, without looking back at the chat.
//
// The harness launches it with ONE block of ONE thread: chase<<<1, 1>>>.
// next[k] holds the index of the next slot to visit. Start at slot 0.
// Follow the chain for `steps` loads, each load needing the answer of the one before.
// Time the loop with clock64(), and write the average ticks per load to *ticks.
// Write the last index to *sink, so that the compiler cannot delete your loop.

__global__ void chase(const int* next, int steps, int* sink, long long* ticks)
{
    int p = 0;
    long long t0 = clock64();
    for(int i = 0; i < steps; i++) {
        p = next[p];
    }
    long long t1 = clock64();
    *sink = p;
    *ticks = (t1 - t0) / steps;
}
