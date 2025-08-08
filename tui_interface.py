import os
import shutil
import time
import sys
import re
from datetime import datetime
from municipal_code_assistant import MunicipalCodeAssistant


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    DIM = '\033[2m'
    BRIGHT_BLUE = '\033[94m\033[1m'
    BRIGHT_GREEN = '\033[92m\033[1m'


class TUIInterface:
    """Terminal User Interface for Municipal Code Assistant"""
    
    def __init__(self, assistant):
        self.assistant = assistant
    
    def get_terminal_width(self):
        """Get terminal width, default to 80 if can't determine"""
        try:
            return shutil.get_terminal_size().columns
        except:
            return 80
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print the main header with ASCII art"""
        width = self.get_terminal_width()
        self.clear_screen()
        
        ascii_art = f"""{Colors.BRIGHT_BLUE}
╔═══════════════════════════════════════════════════════════════════════════════╗
║  ███████╗██╗         ██████╗  █████╗ ███████╗ ██████╗      █████╗ ██╗        ║
║  ██╔════╝██║         ██╔══██╗██╔══██╗██╔════╝██╔═══██╗    ██╔══██╗██║        ║
║  █████╗  ██║         ██████╔╝███████║███████╗██║   ██║    ███████║██║        ║
║  ██╔══╝  ██║         ██╔═══╝ ██╔══██║╚════██║██║   ██║    ██╔══██║██║        ║
║  ███████╗███████╗    ██║     ██║  ██║███████║╚██████╔╝    ██║  ██║██║        ║
║  ╚══════╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚══════╝ ╚═════╝     ╚═╝  ╚═╝╚═╝        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                      {Colors.YELLOW}Municipal Code Assistant{Colors.BRIGHT_BLUE}                           ║
║                    {Colors.DIM}Ask questions about city regulations{Colors.BRIGHT_BLUE}                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}"""
        
        print(ascii_art)
    
    def print_separator(self, char="═", color=Colors.BLUE):
        """Print a separator line across the terminal width"""
        width = self.get_terminal_width()
        print(f"{color}{char * width}{Colors.END}")
    
    def print_status_bar(self, message, status="INFO"):
        """Print a status bar with message"""
        width = self.get_terminal_width()
        status_colors = {
            "INFO": Colors.BLUE,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED,
            "LOADING": Colors.CYAN
        }
        
        color = status_colors.get(status, Colors.BLUE)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        status_text = f"[{timestamp}] {message}"
        padding = width - len(status_text) - 2
        if padding < 0:
            status_text = status_text[:width-5] + "..."
            padding = 0
        
        print(f"{color}┌{'─' * (width-2)}┐")
        print(f"│ {status_text}{' ' * padding}│")
        print(f"└{'─' * (width-2)}┘{Colors.END}")
    
    def print_loading_animation(self, message):
        """Print a loading message with animation"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        for i in range(10):
            sys.stdout.write(f"\r{Colors.CYAN}{spinner[i % len(spinner)]} {message}...{Colors.END}")
            sys.stdout.flush()
            time.sleep(0.1)
        print()
    
    def print_section_box(self, title, content, section_num):
        """Print content in a nice box format"""
        width = self.get_terminal_width()
        max_content_width = width - 4
        
        print(f"\n{Colors.BRIGHT_GREEN}┌─ Section {section_num}: {title} {'─' * max(0, width - len(title) - len(section_num) - 15)}┐{Colors.END}")
        
        # Word wrap content
        words = content.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_content_width - 2:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            padding = max_content_width - len(line)
            print(f"{Colors.GREEN}│ {Colors.END}{line}{' ' * padding} {Colors.GREEN}│{Colors.END}")
        
        print(f"{Colors.BRIGHT_GREEN}└{'─' * (width-2)}┘{Colors.END}")
    
    def highlight_sections_in_text(self, text):
        """Highlight section references in green"""
        pattern = r'\b(Section\s+\d+\.\d+\.\d+(?:\.\d+)*)\b'
        
        def colorize_match(match):
            section_ref = match.group(1)
            return f"{Colors.BRIGHT_GREEN}{section_ref}{Colors.END}"
        
        return re.sub(pattern, colorize_match, text, flags=re.IGNORECASE)
    
    def print_ai_response_box(self, response):
        """Print AI response in a special box with section highlighting"""
        width = self.get_terminal_width()
        max_content_width = width - 4
        
        print(f"\n{Colors.BRIGHT_BLUE}╔{'═' * (width-2)}╗{Colors.END}")
        print(f"{Colors.BRIGHT_BLUE}║{Colors.YELLOW} 🤖 AI ASSISTANT RESPONSE{' ' * (width - 27)}║{Colors.END}")
        print(f"{Colors.BRIGHT_BLUE}╠{'═' * (width-2)}╣{Colors.END}")
        
        highlighted_response = self.highlight_sections_in_text(response)
        
        # Handle word wrapping with ANSI codes
        paragraphs = highlighted_response.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0:
                padding = max_content_width
                print(f"{Colors.BLUE}║ {' ' * padding} ║{Colors.END}")
            
            words = paragraph.split()
            current_line = ""
            current_visible_length = 0
            
            for word in words:
                visible_word = re.sub(r'\x1b\[[0-9;]*m', '', word)
                
                test_visible_length = current_visible_length
                if current_line:
                    test_visible_length += 1
                test_visible_length += len(visible_word)
                
                if test_visible_length <= max_content_width - 2:
                    if current_line:
                        current_line += " "
                        current_visible_length += 1
                    current_line += word
                    current_visible_length += len(visible_word)
                else:
                    if current_line:
                        visible_line = re.sub(r'\x1b\[[0-9;]*m', '', current_line)
                        padding = max_content_width - len(visible_line)
                        print(f"{Colors.BLUE}║ {Colors.END}{current_line}{' ' * padding} {Colors.BLUE}║{Colors.END}")
                    
                    current_line = word
                    current_visible_length = len(visible_word)
            
            if current_line:
                visible_line = re.sub(r'\x1b\[[0-9;]*m', '', current_line)
                padding = max_content_width - len(visible_line)
                print(f"{Colors.BLUE}║ {Colors.END}{current_line}{' ' * padding} {Colors.BLUE}║{Colors.END}")
        
        print(f"{Colors.BRIGHT_BLUE}╚{'═' * (width-2)}╝{Colors.END}")
    
    def get_styled_input(self):
        """Get user input with nice styling"""
        width = self.get_terminal_width()
        
        print("\n" * 2)
        print(f"{Colors.YELLOW}┌─ Your Question {'─' * (width - 18)}┐{Colors.END}")
        print(f"{Colors.YELLOW}│{Colors.END}")
        
        try:
            question = input(f"{Colors.YELLOW}│{Colors.END} {Colors.BRIGHT_GREEN}🤔{Colors.END} ")
            print(f"{Colors.YELLOW}│{Colors.END}")
            print(f"{Colors.YELLOW}└{'─' * (width-2)}┘{Colors.END}")
            print("\n")
            return question
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}└{'─' * (width-2)}┘{Colors.END}")
            print("\n")
            raise
    
    def show_instructions(self):
        """Show usage instructions"""
        width = self.get_terminal_width()
        print(f"\n{Colors.CYAN}┌─ Instructions {'─' * (width - 16)}┐")
        print(f"│ • Ask questions about El Paso municipal code{' ' * (width - 48)}│")
        print(f"│ • Topics: fences, zoning, permits, regulations{' ' * (width - 48)}│")
        print(f"│ • Type 'exit' or 'quit' to end session{' ' * (width - 40)}│")
        print(f"│ • Press Ctrl+C to interrupt at any time{' ' * (width - 41)}│")
        print(f"└{'─' * (width-2)}┘{Colors.END}")
    
    def show_goodbye(self):
        """Show goodbye message"""
        width = self.get_terminal_width()
        print(f"\n{Colors.BRIGHT_BLUE}┌─ Session Complete {'─' * (width - 20)}┐")
        print(f"│ {Colors.GREEN}Thanks for using El Paso Municipal Code Assistant!{' ' * (width - 52)} {Colors.BRIGHT_BLUE}│")
        print(f"└{'─' * (width-2)}┘{Colors.END}")
    
    def run(self):
        """Main TUI loop"""
        self.print_header()
        
        # Initialize assistant
        try:
            self.print_status_bar("Loading El Paso Municipal Code knowledge base...", "LOADING")
            self.print_loading_animation("Initializing AI components")
            self.assistant.initialize()
            self.print_status_bar("Knowledge base loaded successfully!", "SUCCESS")
        except Exception as e:
            self.print_status_bar(f"Failed to initialize system: {e}", "ERROR")
            return
        
        self.show_instructions()
        
        while True:
            try:
                question = self.get_styled_input()
                if question.lower() in ["exit", "quit"]:
                    break
                
                self.print_status_bar("Searching municipal code database...", "LOADING")
                self.print_loading_animation("Analyzing your question")
                
                # Ask the assistant
                result = self.assistant.ask_question(question)
                
                if not result['success']:
                    self.print_status_bar(result['error'], "WARNING")
                    print(f"\n{Colors.YELLOW}💡 Try rephrasing your question or being more specific.{Colors.END}")
                    continue
                
                self.print_status_bar(f"Found {len(result['documents'])} relevant sections", "SUCCESS")
                
                # Display sections
                for i, doc in enumerate(result['documents'], 1):
                    section = doc.metadata.get('section', 'N/A')
                    content = doc.page_content
                    display_content = content[:600] + "..." if len(content) > 600 else content
                    self.print_section_box("Municipal Code", display_content, section)
                
                # Display AI response
                if result['answer']:
                    self.print_status_bar("Generating AI analysis...", "LOADING")
                    self.print_loading_animation("Processing legal text")
                    self.print_ai_response_box(result['answer'])
                elif result['error']:
                    self.print_status_bar(result['error'], "ERROR")
                    print(f"\n{Colors.RED}Please refer to the code sections above for guidance.{Colors.END}")
                
                self.print_separator("─", Colors.DIM)
                print("\n")
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Session interrupted by user.{Colors.END}")
                break
            except Exception as e:
                self.print_status_bar(f"An error occurred: {e}", "ERROR")
        
        self.show_goodbye()