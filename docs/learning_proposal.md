# Engineering Proposal & Real-Time Invariant Specification

## Purpose
This document establishes the architectural foundation and physical guardrails for Task 9: Real-Time WebRTC Direct Peer-to-Peer Audio Pipeline, RFC 3550 Adaptive Jitter Buffer, and Packet Loss Concealment (PLC).

---

## Invariant XI: Real-Time WebRTC P2P Audio Pipeline, Playout Bounding & Zero-Stall Concealment

1. **RFC 3550 Interarrival Jitter Smoothing Invariant**:
   Interarrival transit delay variation must be computed per RFC 3550 Section 6.4.1 using first-order recursive exponential smoothing with $\alpha = 1/16$:
   $$D(i, j) = (R_j - S_j) - (R_i - S_i)$$
   $$J(i) = J(i-1) + \frac{|D(i-1, i)| - J(i-1)}{16}$$
   where $R$ is local arrival time in 16 kHz sample clock ticks and $S$ is the sender RTP timestamp. Calculations must be independent of local system clock offsets.

2. **Strict Playout Latency Bounding Invariant**:
   The adaptive playout buffer depth target must be dynamically bounded within:
   $$D_{\text{target}} \in [16.0, 100.0]\text{ ms}$$
   derived from $3 \cdot J$. Ingress queues must strictly enforce bounded memory capacity ($\le 9$ packets, $\approx 144\text{ ms}$). Stale and overflow packets beyond the maximum delay bound must be shed immediately, guaranteeing that network bursts or processing pauses cannot induce unbounded playout lag.

3. **ITU-T G.711 Appendix I Energy Decay Invariant**:
   Packet loss concealment (PLC) triggered by sequence gaps or buffer underruns must synthesize replacement frames via pitch-synchronous history replication anchored to an unattenuated pitch template, decayed geometrically by:
   $$g(k) = 0.85^k$$
   for the $k$-th consecutive lost frame. Consecutive loss counts must reset immediately upon receipt of a valid packet.

4. **Burst Loss & Pause Re-Synchronization Invariant**:
   Sequence gaps exceeding the buffer capacity ($> 6$ packets / $100\text{ ms}$) indicate an extended network pause or severe burst loss. The jitter buffer must automatically re-synchronize playout to the incoming live packet rather than executing dozens of synthetic PLC cycles, preventing artificial conversation lag.

5. **Monotonic Sequence Unwrapping Invariant**:
   16-bit RTP sequence numbers ($0 \le seq \le 65535$) must be mapped into monotonic 64-bit integer space using modular distance arithmetic relative to the maximum observed sequence number:
   $$\Delta_{\text{seq}} = (seq - (\text{max\_seq} \ \& \ \text{0xFFFF})) \ \& \ \text{0xFFFF}$$
   $$\text{unwrapped} = \begin{cases} \text{max\_seq} + \Delta_{\text{seq}} & \text{if } \Delta_{\text{seq}} < 32768 \\ \text{max\_seq} - (65536 - \Delta_{\text{seq}}) & \text{otherwise} \end{cases}$$
   guaranteeing correct heap sorting across boundary rollovers ($65535 \to 0$).

---

## Rule 13: WebRTC Audio Transport, Adaptive Jitter Buffering & Packet Loss Concealment (RFC 3550 & G.711)

- **RFC 3550 Interarrival Jitter Calculation**: Interarrival transit delay variation must be computed per RFC 3550 Section 6.4.1 using 1st-order recursive exponential smoothing with $\alpha = 1/16$:
  $$D(i, j) = (R_j - S_j) - (R_i - S_i), \quad J(i) = J(i-1) + \frac{|D(i-1, i)| - J(i-1)}{16}$$
  where $R$ is receiver arrival time in sample clock ticks and $S$ is sender timestamp.
- **Adaptive Jitter Buffer Bounding**: The jitter buffer depth must adapt dynamically between a lower bound ($16.0\text{ ms}$, 1 frame hop) and upper bound ($100.0\text{ ms}$, ~6 frames) based on the smoothed interarrival variance ($3 \cdot J$). 16-bit sequence number wraparound ($65535 \to 0$) must be unwrapped to monotonic 64-bit space to prevent sequence sorting anomalies.
- **Packet Loss Concealment (PLC)**: Buffer underruns or missing sequence packets must trigger ITU-T G.711 App. I pitch-synchronous history replication with geometric energy attenuation ($g = 0.85^k$), preventing audible clicks, pops, or audio processing stalls.
