// Phase 0 gate (PLAN.md 0.1, BUGS.md L-01).
//
// This program contains a deliberate heap-buffer-overflow.  Built with
// -DCMAKE_BUILD_TYPE=Debug it MUST abort with an AddressSanitizer report.
//
// If it exits 0, the sanitizers are not actually running, and every assumption
// PLAN.md makes about catching memory bugs during Phase 2 is false.  That is
// exactly the situation L-01 was written to prevent: a sanitizer you *discover*
// is broken while debugging is worse than no sanitizer, because you already
// committed to a plan that assumed it.

#include <cstdio>

int main() {
    std::printf("asan_probe: writing one past the end of a 4-element heap array.
");
    std::printf("EXPECTED: an AddressSanitizer heap-buffer-overflow report, then abort.
");
    std::printf("If you see \"probe survived\" below, the sanitizers are NOT active.
");

    int* p = new int[4];
    p[4] = 1;  // <-- deliberate: one past the end
    const int leaked = p[4];
    delete[] p;

    std::printf("probe survived (value %d) -- SANITIZERS ARE NOT ACTIVE
", leaked);
    return 0;
}
