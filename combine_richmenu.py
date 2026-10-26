"""Render the single-action image-only menu on macOS. Does not contact LINE.

The generated image is committed for use on other platforms. Native AppKit text
layout keeps Thai marks positioned correctly, without an external font dependency.
"""
from pathlib import Path
import platform
import subprocess
import tempfile


def main():
    if platform.system() != 'Darwin':
        raise SystemExit('Use the supplied image-only-richmenu.png, or render this asset on macOS.')
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='rice-menu-') as cache:
        subprocess.run([
            'swift', '-module-cache-path', cache,
            str(root / 'scripts/render-richmenu.swift'),
            str(root / 'image-only-richmenu.png')
        ], check=True)
    print('Created image-only-richmenu.png (2500x843); no LINE settings changed.')


if __name__ == '__main__':
    main()
