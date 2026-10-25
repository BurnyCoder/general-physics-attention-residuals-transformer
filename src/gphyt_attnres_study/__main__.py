"""Global: module-style CLI for portable long-running processes.

Local: avoids holding a Windows console-script launcher during environment maintenance.
Source: https://docs.python.org/3/library/__main__.html.
"""
from .pipeline import main  # Local: reuse the single readable public phase dispatcher.

main()  # Global: `python -m gphyt_attnres_study` and the console script have identical semantics.
