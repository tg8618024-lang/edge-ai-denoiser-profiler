// Edge AI Voice - WebRTC MediaDevices Shim
// Intercepts getUserMedia to route microphone audio through real-time denoiser filter.

(function () {
  console.log("[Edge AI Denoiser] WebRTC Shim loaded into page execution context.");

  let activeConfig = {
    active: true,
    mode: "neural",
    intensity: 100,
  };

  // Listen for config changes from content script
  window.addEventListener("message", (event) => {
    if (event.source === window && event.data && event.data.source === "EDGE_AI_EXTENSION") {
      if (event.data.type === "CONFIG_UPDATE") {
        activeConfig = Object.assign(activeConfig, event.data.config);
        console.log("[Edge AI Denoiser] Live config updated:", activeConfig);
      }
    }
  });

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    console.warn("[Edge AI Denoiser] navigator.mediaDevices.getUserMedia not available.");
    return;
  }

  const origGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

  navigator.mediaDevices.getUserMedia = async function (constraints) {
    console.log("[Edge AI Denoiser] Intercepting getUserMedia constraints:", constraints);

    // If audio is not requested or denoiser disabled, return standard stream
    if (!constraints || !constraints.audio || !activeConfig.active) {
      return origGetUserMedia(constraints);
    }

    try {
      const rawStream = await origGetUserMedia(constraints);
      const audioTracks = rawStream.getAudioTracks();

      if (!audioTracks || audioTracks.length === 0) {
        return rawStream;
      }

      console.log("[Edge AI Denoiser] Setting up Web Audio processing graph for microphone...");

      // Initialize high-performance Web Audio pipeline
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
        latencyHint: "interactive",
      });

      const source = audioCtx.createMediaStreamSource(rawStream);
      const destination = audioCtx.createMediaStreamDestination();

      // Bandpass / dynamic voice isolation node (in-browser zero-overhead filter)
      const highpass = audioCtx.createBiquadFilter();
      highpass.type = "highpass";
      highpass.frequency.value = 85.0; // Cut sub-rumble

      const lowpass = audioCtx.createBiquadFilter();
      lowpass.type = "lowpass";
      lowpass.frequency.value = 7500.0; // Cut ultrasonic hiss

      const compressor = audioCtx.createDynamicsCompressor();
      compressor.threshold.setValueAtTime(-24, audioCtx.currentTime);
      compressor.knee.setValueAtTime(30, audioCtx.currentTime);
      compressor.ratio.setValueAtTime(12, audioCtx.currentTime);
      compressor.attack.setValueAtTime(0.003, audioCtx.currentTime);
      compressor.release.setValueAtTime(0.25, audioCtx.currentTime);

      source.connect(highpass);
      highpass.connect(lowpass);
      lowpass.connect(compressor);
      compressor.connect(destination);

      // Create hybrid stream containing denoised audio + original video
      const processedStream = new MediaStream();
      processedStream.addTrack(destination.stream.getAudioTracks()[0]);

      rawStream.getVideoTracks().forEach((track) => {
        processedStream.addTrack(track);
      });

      console.log("[Edge AI Denoiser] Clean audio graph active for call!");
      return processedStream;
    } catch (err) {
      console.error("[Edge AI Denoiser] Audio graph fallback to raw microphone:", err);
      return origGetUserMedia(constraints);
    }
  };
})();
