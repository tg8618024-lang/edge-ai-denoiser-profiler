import 'package:flutter/material.dart';

/// Real-time scrollable telemetry event log list.
/// Used to demonstrate dynamic long list rendering and scrollUntilVisible testing.
class TelemetryEventList extends StatelessWidget {
  final List<String> events;

  const TelemetryEventList({
    super.key,
    required this.events,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 3,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    const Icon(Icons.list_alt, color: Colors.tealAccent),
                    const SizedBox(width: 8),
                    Text(
                      'Real-Time Frame Stream Log',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                Text(
                  '${events.length} frames',
                  style: const TextStyle(fontSize: 12, color: Colors.white54),
                ),
              ],
            ),
            const Divider(),
            SizedBox(
              height: 220,
              child: ListView.builder(
                key: const Key('telemetry_event_list'),
                itemCount: events.length,
                itemBuilder: (context, index) {
                  final event = events[index];
                  return Container(
                    key: Key('event_item_$index'),
                    padding: const EdgeInsets.symmetric(vertical: 6.0, horizontal: 8.0),
                    margin: const EdgeInsets.symmetric(vertical: 2.0),
                    decoration: BoxDecoration(
                      color: index.isEven ? Colors.black12 : Colors.white.withOpacity(0.04),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.check_circle_outline, color: Colors.greenAccent, size: 14),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            event,
                            style: const TextStyle(fontSize: 12, fontFamily: 'monospace'),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
