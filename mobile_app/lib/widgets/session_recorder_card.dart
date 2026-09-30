import 'package:flutter/material.dart';

/// Multi-track Studio Session Recorder & Telemetry Exporter card.
class SessionRecorderCard extends StatelessWidget {
  final bool isRecording;
  final bool isExported;
  final VoidCallback onRecordToggled;
  final VoidCallback onExportTelemetry;

  const SessionRecorderCard({
    super.key,
    required this.isRecording,
    required this.isExported,
    required this.onRecordToggled,
    required this.onExportTelemetry,
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
              children: [
                const Icon(Icons.album_outlined, color: Colors.redAccent),
                const SizedBox(width: 8),
                Text(
                  'Multi-Track Session Studio',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const Divider(),
            const SizedBox(height: 6),

            Row(
              children: [
                // Record Button
                ElevatedButton.icon(
                  key: const Key('record_session_btn'),
                  onPressed: onRecordToggled,
                  icon: Icon(
                    isRecording ? Icons.stop : Icons.fiber_manual_record,
                    color: isRecording ? Colors.white : Colors.redAccent,
                  ),
                  label: Text(isRecording ? 'Stop Recording' : 'Record Session'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isRecording ? Colors.redAccent : Colors.white12,
                    foregroundColor: Colors.white,
                  ),
                ),
                const SizedBox(width: 12),
                // Export Button
                OutlinedButton.icon(
                  key: const Key('export_telemetry_btn'),
                  onPressed: onExportTelemetry,
                  icon: const Icon(Icons.file_download_outlined),
                  label: const Text('Export Telemetry (CSV/JSON)'),
                ),
              ],
            ),
            const SizedBox(height: 10),

            // Export status banner if triggered
            if (isExported)
              Container(
                key: const Key('telemetry_export_status'),
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.greenAccent.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: Colors.greenAccent),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.check_circle_outline, color: Colors.greenAccent, size: 16),
                    SizedBox(width: 8),
                    Text(
                      'Telemetry & WAV Multi-Track Bundle Exported Successfully!',
                      style: TextStyle(fontSize: 12, color: Colors.greenAccent, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
              ),

            const SizedBox(height: 6),
            const Text(
              'Exports 4 synchronous tracks: Clean WAV (16kHz PCM), Raw Noisy WAV, Subtracted Delta Floor, and SRT Subtitles.',
              style: TextStyle(fontSize: 11, color: Colors.white38),
            ),
          ],
        ),
      ),
    );
  }
}
