import 'package:flutter/material.dart';
import '../models/denoiser_state.dart';

/// Multi-precision Quantization Comparison Mode selector (FP32, FP16, INT8).
/// Displays memory footprint, compression factor, and SQNR quality metrics.
class PrecisionSelectorCard extends StatelessWidget {
  final PrecisionMode selectedPrecision;
  final ValueChanged<PrecisionMode> onPrecisionSelected;

  const PrecisionSelectorCard({
    super.key,
    required this.selectedPrecision,
    required this.onPrecisionSelected,
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
                const Icon(Icons.memory, color: Colors.purpleAccent),
                const SizedBox(width: 8),
                Text(
                  'Multi-Precision Engine',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const Divider(),
            const SizedBox(height: 8),

            // Precision Mode Selection Chips
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _buildChoiceChip(
                  key: const Key('precision_chip_fp32'),
                  mode: PrecisionMode.fp32,
                  subtitle: '165.8 KB (Full)',
                ),
                _buildChoiceChip(
                  key: const Key('precision_chip_fp16'),
                  mode: PrecisionMode.fp16,
                  subtitle: '82.9 KB (Half)',
                ),
                _buildChoiceChip(
                  key: const Key('precision_chip_int8'),
                  mode: PrecisionMode.int8,
                  subtitle: '42.6 KB (SIMD)',
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Quantization Hardware Details
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black26,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildDetailItem(
                    label: 'RAM Footprint',
                    value: '${selectedPrecision.memoryKb} KB',
                    keyName: 'precision_memory_text',
                    color: Colors.greenAccent,
                  ),
                  _buildDetailItem(
                    label: 'RAM Compression',
                    value: selectedPrecision.compressionRate,
                    keyName: 'precision_compression_text',
                    color: Colors.cyanAccent,
                  ),
                  _buildDetailItem(
                    label: 'Signal SQNR',
                    value: selectedPrecision.sqnrDb,
                    keyName: 'precision_sqnr_text',
                    color: Colors.amberAccent,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChoiceChip({
    required Key key,
    required PrecisionMode mode,
    required String subtitle,
  }) {
    final isSelected = selectedPrecision == mode;
    return ChoiceChip(
      key: key,
      selected: isSelected,
      label: Column(
        children: [
          Text(mode.label, style: const TextStyle(fontWeight: FontWeight.bold)),
          Text(subtitle, style: const TextStyle(fontSize: 10)),
        ],
      ),
      selectedColor: Colors.purpleAccent.withOpacity(0.3),
      onSelected: (selected) {
        if (selected) {
          onPrecisionSelected(mode);
        }
      },
    );
  }

  Widget _buildDetailItem({
    required String label,
    required String value,
    required String keyName,
    required Color color,
  }) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.white54)),
        const SizedBox(height: 2),
        Text(
          value,
          key: Key(keyName),
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ],
    );
  }
}
