import 'package:flutter/material.dart';
import 'screens/denoiser_dashboard_screen.dart';

void main() {
  runApp(const EdgeAiDenoiserApp());
}

/// Root Application Widget for Edge AI Audio Denoiser & Profiler.
class EdgeAiDenoiserApp extends StatelessWidget {
  const EdgeAiDenoiserApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Edge AI Audio Denoiser & Profiler',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0F141C),
        colorScheme: const ColorScheme.dark(
          primary: Colors.greenAccent,
          secondary: Colors.cyanAccent,
          surface: Color(0xFF161E2E),
        ),
        cardTheme: const CardTheme(
          color: Color(0xFF161E2E),
          elevation: 2,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF111827),
          elevation: 0,
        ),
      ),
      home: const DenoiserDashboardScreen(),
    );
  }
}
