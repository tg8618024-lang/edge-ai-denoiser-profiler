import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_denoiser_mobile/models/denoiser_state.dart';
import 'package:edge_ai_denoiser_mobile/widgets/denoiser_control_card.dart';

void main() {
  group('DenoiserControlCard Component Tests', () {
    testWidgets('Renders initial static state with Denoised active and 85% suppression', (WidgetTester tester) async {
      // 1. Build the widget wrapped in MaterialApp
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: DenoiserControlCard(
              isDenoisingActive: true,
              isMuted: false,
              suppressionIntensity: 0.85,
              selectedProfile: NoiseProfile.white,
              onDenoisingToggled: (_) {},
              onMuteToggled: () {},
              onSuppressionChanged: (_) {},
              onProfileChanged: (_) {},
            ),
          ),
        ),
      );

      // 2. Verify initial UI elements and texts
      expect(find.text('Neural Denoiser Engine'), findsOneWidget);
      expect(find.text('Denoised (Clean Voice)'), findsOneWidget);
      expect(find.text('85%'), findsOneWidget);
      expect(find.byKey(const Key('denoiser_toggle_switch')), findsOneWidget);
      expect(find.byKey(const Key('mute_toggle_button')), findsOneWidget);
      expect(find.byKey(const Key('suppression_slider')), findsOneWidget);
      expect(find.byKey(const Key('noise_profile_dropdown')), findsOneWidget);
      expect(find.byIcon(Icons.volume_up), findsOneWidget);
      expect(find.byIcon(Icons.volume_off), findsNothing);
    });

    testWidgets('Tapping denoiser toggle switch invokes onDenoisingToggled callback', (WidgetTester tester) async {
      bool? toggledValue;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return DenoiserControlCard(
                  isDenoisingActive: true,
                  isMuted: false,
                  suppressionIntensity: 0.85,
                  selectedProfile: NoiseProfile.white,
                  onDenoisingToggled: (val) {
                    toggledValue = val;
                  },
                  onMuteToggled: () {},
                  onSuppressionChanged: (_) {},
                  onProfileChanged: (_) {},
                );
              },
            ),
          ),
        ),
      );

      // Locate switch and simulate tap
      final switchFinder = find.byKey(const Key('denoiser_toggle_switch'));
      expect(switchFinder, findsOneWidget);

      await tester.tap(switchFinder);
      await tester.pump();

      // Verify callback triggered with inverted value
      expect(toggledValue, isFalse);
    });

    testWidgets('Tapping mute toggle button switches icon from volume_up to volume_off', (WidgetTester tester) async {
      bool isMuted = false;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return DenoiserControlCard(
                  isDenoisingActive: true,
                  isMuted: isMuted,
                  suppressionIntensity: 0.85,
                  selectedProfile: NoiseProfile.white,
                  onDenoisingToggled: (_) {},
                  onMuteToggled: () {
                    setState(() {
                      isMuted = !isMuted;
                    });
                  },
                  onSuppressionChanged: (_) {},
                  onProfileChanged: (_) {},
                );
              },
            ),
          ),
        ),
      );

      // Verify initial unmuted icon
      expect(find.byIcon(Icons.volume_up), findsOneWidget);
      expect(find.byIcon(Icons.volume_off), findsNothing);

      // Tap the mute button
      await tester.tap(find.byKey(const Key('mute_toggle_button')));
      await tester.pump();

      // Verify icon changed to volume_off
      expect(find.byIcon(Icons.volume_off), findsOneWidget);
      expect(find.byIcon(Icons.volume_up), findsNothing);
      expect(isMuted, isTrue);
    });

    testWidgets('Dragging suppression slider triggers onSuppressionChanged with new value', (WidgetTester tester) async {
      double suppression = 0.50;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return DenoiserControlCard(
                  isDenoisingActive: true,
                  isMuted: false,
                  suppressionIntensity: suppression,
                  selectedProfile: NoiseProfile.white,
                  onDenoisingToggled: (_) {},
                  onMuteToggled: () {},
                  onSuppressionChanged: (newVal) {
                    setState(() {
                      suppression = newVal;
                    });
                  },
                  onProfileChanged: (_) {},
                );
              },
            ),
          ),
        ),
      );

      expect(find.text('50%'), findsOneWidget);

      // Drag the slider horizontally to the right
      final sliderFinder = find.byKey(const Key('suppression_slider'));
      await tester.drag(sliderFinder, const Offset(100.0, 0.0));
      await tester.pumpAndSettle();

      // Verify suppression intensity increased
      expect(suppression, greaterThan(0.50));
    });

    testWidgets('Selecting a noise profile from dropdown invokes onProfileChanged callback', (WidgetTester tester) async {
      NoiseProfile selected = NoiseProfile.white;

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: StatefulBuilder(
              builder: (context, setState) {
                return DenoiserControlCard(
                  isDenoisingActive: true,
                  isMuted: false,
                  suppressionIntensity: 0.85,
                  selectedProfile: selected,
                  onDenoisingToggled: (_) {},
                  onMuteToggled: () {},
                  onSuppressionChanged: (_) {},
                  onProfileChanged: (newProfile) {
                    setState(() {
                      if (newProfile != null) selected = newProfile;
                    });
                  },
                );
              },
            ),
          ),
        ),
      );

      // Verify initial selected profile display
      expect(find.text('White Noise'), findsOneWidget);

      // Tap dropdown to open popup menu
      final dropdownFinder = find.byKey(const Key('noise_profile_dropdown'));
      await tester.tap(dropdownFinder);
      await tester.pumpAndSettle();

      // Tap 'Drone / Fan Hum' menu item in the popup menu
      final itemFinder = find.text('Drone / Fan Hum').last;
      await tester.tap(itemFinder);
      await tester.pumpAndSettle();

      // Verify callback updated selected profile
      expect(selected, equals(NoiseProfile.drone));
    });

    testWidgets('Displays Bypass label when isDenoisingActive is false', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: DenoiserControlCard(
              isDenoisingActive: false,
              isMuted: false,
              suppressionIntensity: 0.85,
              selectedProfile: NoiseProfile.drone,
              onDenoisingToggled: (_) {},
              onMuteToggled: () {},
              onSuppressionChanged: (_) {},
              onProfileChanged: (_) {},
            ),
          ),
        ),
      );

      expect(find.text('Bypass (Noisy Input)'), findsOneWidget);
      expect(find.text('Denoised (Clean Voice)'), findsNothing);
      expect(find.text('Raw acoustic passthrough'), findsOneWidget);
    });
  });
}
