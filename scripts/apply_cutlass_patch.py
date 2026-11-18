from pathlib import Path
import subprocess
import sys


def apply_cutlass_patch(cutlass_dir: Path) -> None:
    patch = Path(__file__).with_name('cutlass-windows-host.patch').resolve()
    command = ['git', 'apply']
    reverse = subprocess.run(
        [*command, '--reverse', '--check', str(patch)],
        cwd=cutlass_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if reverse.returncode == 0:
        print('CUTLASS Windows host patch is already applied', flush=True)
        return

    subprocess.run(
        [*command, '--check', str(patch)], cwd=cutlass_dir, check=True)
    subprocess.run([*command, str(patch)], cwd=cutlass_dir, check=True)
    print('Applied CUTLASS Windows host patch', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: apply_cutlass_patch.py <CUTLASS_SOURCE_DIR>')
    apply_cutlass_patch(Path(sys.argv[1]).resolve())
