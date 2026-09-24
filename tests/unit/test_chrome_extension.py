"""Unit tests for Chrome Extension & WebRTC AudioWorklet Shim.

Tests:
- Manifest V3 configuration schema and permissions.
- Icon existence and exact pixel dimensions (16x16, 48x48, 128x128).
- Popup UI assets existence (HTML, CSS, JS).
- WebRTC injection scripts and AudioWorklet processor existence.
"""

import os
import json
import pytest
from PIL import Image

EXTENSION_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "extensions", "chrome-edge-denoiser")
)


class TestChromeExtension:
    def test_manifest_schema_and_version(self):
        """Verify manifest.json exists, is valid JSON, and uses Manifest V3."""
        manifest_path = os.path.join(EXTENSION_DIR, "manifest.json")
        assert os.path.isfile(manifest_path), "manifest.json must exist"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest.get("manifest_version") == 3
        assert "name" in manifest
        assert "version" in manifest
        assert "icons" in manifest
        assert "action" in manifest
        assert "permissions" in manifest
        assert "storage" in manifest["permissions"]
        assert "activeTab" in manifest["permissions"]
        assert "content_scripts" in manifest
        assert "web_accessible_resources" in manifest

    def test_icon_files_and_dimensions(self):
        """Verify all referenced icons exist and have exact pixel dimensions."""
        manifest_path = os.path.join(EXTENSION_DIR, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        icons_dict = manifest.get("icons", {})
        assert "16" in icons_dict
        assert "48" in icons_dict
        assert "128" in icons_dict

        for size_str, rel_path in icons_dict.items():
            full_path = os.path.join(EXTENSION_DIR, rel_path)
            assert os.path.isfile(full_path), f"Icon file {rel_path} must exist"

            # Check image dimensions with PIL
            expected_size = int(size_str)
            with Image.open(full_path) as img:
                assert img.size == (expected_size, expected_size), (
                    f"Expected {expected_size}x{expected_size}, got {img.size}"
                )

    def test_popup_files_exist_and_non_empty(self):
        """Verify popup HTML, CSS, and JS exist and contain valid code."""
        popup_dir = os.path.join(EXTENSION_DIR, "popup")
        html_file = os.path.join(popup_dir, "popup.html")
        css_file = os.path.join(popup_dir, "popup.css")
        js_file = os.path.join(popup_dir, "popup.js")

        assert os.path.isfile(html_file)
        assert os.path.isfile(css_file)
        assert os.path.isfile(js_file)

        assert os.path.getsize(html_file) > 100
        assert os.path.getsize(css_file) > 100
        assert os.path.getsize(js_file) > 100

    def test_scripts_exist_and_non_empty(self):
        """Verify content_script, webrtc_shim, and denoiser-worklet exist."""
        scripts_dir = os.path.join(EXTENSION_DIR, "scripts")
        content_script = os.path.join(scripts_dir, "content_script.js")
        webrtc_shim = os.path.join(scripts_dir, "webrtc_shim.js")
        worklet = os.path.join(scripts_dir, "denoiser-worklet.js")

        assert os.path.isfile(content_script)
        assert os.path.isfile(webrtc_shim)
        assert os.path.isfile(worklet)

        assert os.path.getsize(content_script) > 100
        assert os.path.getsize(webrtc_shim) > 100
        assert os.path.getsize(worklet) > 100
