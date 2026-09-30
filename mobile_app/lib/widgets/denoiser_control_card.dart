import 'package:flutter/material.dart';
import '../models/denoiser_state.dart';

/// Interactive audio denoiser control panel.
/// Allows toggling between Clean Denoised Output and Noisy Bypass,
/// adjusting suppression intensity, muting output, and selecting noise profiles.
class DenoiserControlCard extends StatelessWidget {
  final bool isDenoisingActive;
  final bool isMuted;
  final double suppressionIntensity;
  final NoiseProfile selectedProfile;
  final ValueChanged<bool> onDenoisingToggled;
  final VoidCallback onMuteToggled;
  final ValueChanged<double> onSuppressionChanged;
  final ValueChanged<NoiseProfile?> onProfileChanged;

  const DenoiserControlCard({
    super.key,
    required this.isDenoisingActive,
    required this.isMuted,
    required this.suppressionIntensity,
    required this.selectedProfile,
    required this.onDenoisingToggled,
    required this.onMuteToggled,
    required this.onSuppressionChanged,
    required this.onProfileChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final statusText = isDenoisingActive
        ? 'Denoised (Clean Voice)'
        : 'Bypass (Noisy Input)';
    final suppressionPct = (suppressionIntensity * 100).round();

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
                    Icon(
                      isDenoisingActive
                          ? Icons.noise_aware
                          : Icons.noise_control_off,
                      color: isDenoisingActive ? Colors.greenAccent : Colors.orangeAccent,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Neural Denoiser Engine',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                IconButton(
                  key: const Key('mute_toggle_button'),
                  icon: Icon(
                    isMuted ? Icons.volume_off : Icons.volume_up,
                    color: isMuted ? Colors.redAccent : Colors.white70,
                  ),
                  tooltip: isMuted ? 'Unmute Audio' : 'Mute Audio',
                  onPressed: onMuteToggled,
                ),
              ],
            ),
            const Divider(),
            // A/B Denoising Toggle Switch
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      statusText,
                      style: theme.textTheme.bodyLarge?.copyWith(
                        color: isDenoisingActive ? Colors.greenAccent : Colors.orangeAccent,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      isDenoisingActive
                          ? 'GRU-MaskNet & Wiener filtering active'
                          : 'Raw acoustic passthrough',
                      style: theme.textTheme.bodySmall?.copyWith(color: Colors.white54),
                    ),
                  ],
                ),
                Switch(
                  key: const Key('denoiser_toggle_switch'),
                  value: isDenoisingActive,
                  activeColor: Colors.greenAccent,
                  onChanged: onDenoisingToggled,
                ),
              ],
            ),
            const SizedBox(height: 16),
            // Suppression Intensity Slider
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'Suppression Intensity',
                  style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500),
                ),
                Text(
                  '$suppressionPct%',
                  key: const Key('suppression_percentage_text'),
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.cyanAccent,
                  ),
                ),
              ],
            ),
            Slider(
              key: const Key('suppression_slider'),
              value: suppressionIntensity,
              min: 0.0,
              max: 1.0,
              divisions: 20,
              activeColor: Colors.cyanAccent,
              onChanged: onSuppressionChanged,
            ),
            const SizedBox(height: 12),
            // Noise Profile Dropdown
            Text(
              'Acoustic Noise Benchmark Profile',
              style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.w500),
            ),
            const SizedBox(height: 6),
            DropdownButtonFormField<NoiseProfile>(
              key: const Key('noise_profile_dropdown'),
              value: selectedProfile,
              decoration: InputDecoration(
                contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
              ),
              items: NoiseProfile.values.map((profile) {
                return DropdownMenuItem<NoiseProfile>(
                  value: profile,
                  child: Text(profile.displayName),
                );
              }).toList(),
              onChanged: onProfileChanged,
            ),
          ],
        ),
      ),
    );
  }
}
