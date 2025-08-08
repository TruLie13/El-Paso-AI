#!/usr/bin/env python3
"""
El Paso Municipal Code Assistant
Entry point for the application
"""

from municipal_code_assistant import MunicipalCodeAssistant
from tui_interface import TUIInterface


def main():
    """Main entry point"""
    # Create assistant and TUI
    assistant = MunicipalCodeAssistant(db_path="chroma_db")
    tui = TUIInterface(assistant)
    
    # Run the interface
    tui.run()


if __name__ == "__main__":
    main()