"""Unified GROMACS discovery, version inspection, and execution wrapper."""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("conformatlas.gromacs")


class GromacsRunner:
    """Robust abstraction for discovering and executing GROMACS commands."""

    def __init__(self, gmx_command: str | Path | None = None):
        self.gmx_path = self.locate_gmx(gmx_command)
        self.version = self.detect_version()

    @staticmethod
    def locate_gmx(custom_path: str | Path | None = None) -> Path:
        """Locate GROMACS binary: custom path -> 'gmx' -> 'gmx_mpi'."""
        if custom_path:
            p = Path(custom_path).expanduser().resolve()
            if p.is_file() and os.access(p, os.X_OK):
                return p
            which_p = shutil.which(str(custom_path))
            if which_p:
                return Path(which_p).resolve()
            raise FileNotFoundError(f"Specified GROMACS executable not found or not executable: {custom_path}")

        # Check standard names
        for candidate in ["gmx", "gmx_mpi"]:
            found = shutil.which(candidate)
            if found:
                return Path(found).resolve()

        raise RuntimeError(
            "GROMACS binary ('gmx' or 'gmx_mpi') not found in PATH. "
            "Please ensure GROMACS is loaded or pass --gmx-command."
        )

    def detect_version(self) -> str:
        """Inspect GROMACS version via 'gmx --version'."""
        try:
            res = subprocess.run(
                [str(self.gmx_path), "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = res.stdout or res.stderr
            match = re.search(r"GROMACS version:\s*([^\n\r]+)", output, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            # Alternative match
            match2 = re.search(r"GROMACS\s*-\s*([0-9\.\-A-Za-z]+)", output)
            if match2:
                return match2.group(1).strip()
            return "Unknown (executable detected)"
        except Exception as e:
            logger.warning(f"Could not determine GROMACS version: {e}")
            return "Unknown"

    def run(
        self,
        subcommand: str,
        args: list[str],
        input_text: str | None = None,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Execute a GROMACS command: [gmx_path, subcommand] + args.
        
        Parameters
        ----------
        subcommand : str
            GROMACS tool, e.g., 'covar', 'anaeig', 'sham', 'make_ndx', 'trjconv'.
        args : List[str]
            Command-line arguments passed to the tool.
        input_text : Optional[str]
            Input piped to standard input.
        cwd : Optional[Path]
            Working directory for execution.
        check : bool
            Whether to raise RuntimeError on non-zero exit code.

        Returns
        -------
        subprocess.CompletedProcess
            Captured process execution result.
        """
        full_cmd = [str(self.gmx_path), subcommand] + args
        cmd_str = " ".join(full_cmd)
        if cwd:
            logger.debug(f"[CWD: {cwd}] Executing: {cmd_str}")
        else:
            logger.debug(f"Executing: {cmd_str}")

        if input_text:
            logger.debug(f"Stdin input:\n{input_text.strip()}")

        res = subprocess.run(
            full_cmd,
            input=input_text,
            text=True,
            capture_output=True,
            cwd=cwd,
        )

        if res.returncode != 0:
            err_msg = (
                f"GROMACS command '{subcommand}' failed with code {res.returncode}.\n"
                f"Command: {cmd_str}\n"
            )
            if res.stderr:
                err_msg += f"Stderr:\n{res.stderr.strip()}\n"
            if res.stdout:
                err_msg += f"Stdout:\n{res.stdout.strip()[-1000:]}\n"
            logger.error(err_msg)
            if check:
                raise RuntimeError(err_msg)

        return res
