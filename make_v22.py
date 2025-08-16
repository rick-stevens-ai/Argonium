#!/usr/bin/env python3
# make_v19.py - Question Generation Script
# 
# This version is based on make_v18.py with bulk/batch API support removed
# and enhanced exit sequence with comprehensive statistics summary.

import os
import sys
import json
import re
import time
import random
from openai import OpenAI
import PyPDF2
import spacy
import yaml
import argparse
import pathlib
import signal
import hashlib
import curses
import threading
import queue
import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional, Set
from enum import Enum
from tqdm import tqdm

# Global variables for state tracking
_file_map = {}
_chunk_map = {}
_total_files = 0
_processed_files = 0
_total_chunks = 0
_processed_chunks = 0
_extracted_chunks = 0
_exit_requested = False
_log_queue = queue.Queue()  # Queue for log messages
_use_split_screen = True  # Whether to use the split-screen UI
_start_time = time.time()  # Global start time for tracking elapsed time
_active_workers = 0  # Track active worker processes or threads
_max_workers = None  # Number of worker processes to use (set from args)
_result_queue = None  # Queue for workers to return results
_worker_pool = None  # Worker pool for parallel processing
# Lock for updating shared counters
_counter_lock = threading.Lock()
_error_log_file = None  # File to log errors to
_max_error_threshold = 200  # Maximum number of errors before stopping the program
_openai_client = None  # OpenAI client instance


class TerminalUI:
    """
    Class to manage a split-screen terminal UI with curses.
    The top pane shows global progress statistics and the bottom pane shows scrolling logs.
    """
    def __init__(self):
        self.screen = None
        self.top_pane = None
        self.bottom_pane = None
        self.top_height = 20  # Height of the top stats pane - increased to accommodate error statistics
        self.max_log_lines = 1000  # Maximum number of log lines to keep
        self.log_lines = []
        self.log_position = 0
        self.running = False
        self.ui_thread = None
        self.curses_enabled = False
        self.question_type = None  # Store the question type for display
        
        # Current statistics to display
        self.stats = {
            "files_processed": 0,
            "total_files": 0,
            "chunks_extracted": 0,
            "chunks_processed": 0,
            "total_chunks": 0,
            "questions_generated": 0,
            "success_rate": 0.0,
            "completion_percentage": 0.0,
            "file_percentage": 0.0,  # Add explicit file percentage tracking
            "elapsed_time": 0,
            "eta": "Unknown",
            "avg_chunk_time": "Unknown",
            "current_file": "",
            "current_chunk": "",
            "status_message": "Initializing...",
            "active_workers": 0,
            "max_workers": 0,
            # Error counters
            "error_file_processing": 0,    # Errors processing files
            "error_chunk_extraction": 0,   # Errors extracting chunks
            "error_chunk_reading": 0,      # Errors reading chunks
            "error_summarizing": 0,        # Errors summarizing chunks
            "error_question_gen": 0,       # Errors generating questions
            "error_question_eval": 0,      # Errors evaluating questions
            "error_api": 0,                # API errors
            "error_other": 0,              # Other miscellaneous errors
            "low_score_questions": 0,      # Questions that scored too low
            "total_errors": 0,             # Total of all errors
            "model_name": "Unknown",       # Current AI model being used
        }
    
    def start(self):
        """Start the terminal UI in a separate thread."""
        global _use_split_screen
        if not sys.stdout.isatty() or os.environ.get('TERM') == 'dumb':
            # Terminal doesn't support curses
            self.curses_enabled = False
            _use_split_screen = False
            return False
            
        try:
            self.curses_enabled = True
            self.running = True
            self.ui_thread = threading.Thread(target=self._run_ui, daemon=True)
            self.ui_thread.start()
            return True
        except Exception as e:
            print(f"Error starting terminal UI: {e}")
            self.curses_enabled = False
            _use_split_screen = False
            return False
    
    def stop(self):
        """Stop the terminal UI thread."""
        self.running = False
        if self.ui_thread:
            try:
                self.ui_thread.join(timeout=1.0)
            except:
                pass  # Thread may already be dead
    
    def _run_ui(self):
        """Main loop for the UI thread."""
        global _use_split_screen
        try:
            # Initialize curses
            curses.wrapper(self._curses_main)
        except Exception as e:
            # Fall back to normal output if curses fails
            self.curses_enabled = False
            _use_split_screen = False
            print(f"Error in terminal UI: {e}")
    
    def _curses_main(self, stdscr):
        """Main curses function that sets up the UI."""
        # Set up colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # Success
        curses.init_pair(2, curses.COLOR_RED, -1)    # Error
        curses.init_pair(3, curses.COLOR_YELLOW, -1) # Warning
        curses.init_pair(4, curses.COLOR_CYAN, -1)   # Info
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header
        
        # Hide cursor
        curses.curs_set(0)
        
        # Store screen reference
        self.screen = stdscr
        
        # Get screen dimensions
        height, width = self.screen.getmaxyx()
        
        # Create panes
        self.top_pane = curses.newwin(self.top_height, width, 0, 0)
        self.bottom_pane = curses.newwin(height - self.top_height - 1, width, self.top_height + 1, 0)
        
        # Draw divider line
        divider = curses.newwin(1, width, self.top_height, 0)
        divider.bkgd("-", curses.color_pair(5))
        divider.refresh()
        
        # Configure bottom pane for scrolling
        self.bottom_pane.scrollok(True)
        self.bottom_pane.idlok(True)
        
        # Main loop
        while self.running:
            # Process any new log messages
            self._process_logs()
            
            # Update the display
            self._update_display()
            
            # Sleep briefly to reduce CPU usage
            time.sleep(0.1)
    
    def _process_logs(self):
        """Process any new log messages from the queue."""
        # Get all available log messages
        try:
            while not _log_queue.empty():
                try:
                    log_msg = _log_queue.get_nowait()
                    # Add to log lines, with timestamp
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    
                    # Format message based on level if present
                    if isinstance(log_msg, tuple) and len(log_msg) == 2:
                        msg_text, level = log_msg
                        if level == "ERROR":
                            formatted_msg = f"[{timestamp}] [ERROR] {msg_text}"
                        elif level == "WARNING":
                            formatted_msg = f"[{timestamp}] [WARNING] {msg_text}"
                        else:  # INFO or any other level
                            formatted_msg = f"[{timestamp}] {msg_text}"
                    else:  # Simple string message
                        formatted_msg = f"[{timestamp}] {log_msg}"
                    
                    # Add to log lines
                    self.log_lines.append(formatted_msg)
                    
                    # Trim log if needed
                    if len(self.log_lines) > self.max_log_lines:
                        self.log_lines = self.log_lines[-self.max_log_lines:]
                    
                    # Auto-scroll to the bottom if we were already at the bottom
                    if self.log_position == len(self.log_lines) - 2 or self.log_position >= len(self.log_lines) - 1:
                        self.log_position = len(self.log_lines) - 1
                        
                    # Mark as done
                    _log_queue.task_done()
                except Exception as e:
                    # Safely handle any errors in processing a log message
                    try:
                        self.log_lines.append(f"[{time.strftime('%H:%M:%S', time.localtime())}] [ERROR] Log processing error: {str(e)}")
                        _log_queue.task_done()
                    except:
                        pass  # Ultimate fallback
        except queue.Empty:
            pass  # No more messages
    
    def _update_display(self):
        """Update both panes with the latest information."""
        if not self.screen:
            return
            
        try:
            # Get current terminal dimensions
            height, width = self.screen.getmaxyx()
            
            # Sanity check for terminal size
            if height < 10 or width < 20:  # Terminal too small to be usable
                # Draw simple message if terminal is too small
                try:
                    self.screen.clear()
                    self.screen.addstr(0, 0, "Term too small")
                    self.screen.refresh()
                except curses.error:
                    pass  # Even that might fail if terminal is extremely small
                return
            
            # Resize panes if terminal dimensions changed
            try:
                if self.top_pane.getmaxyx()[1] != width:
                    # Determine appropriate top pane height based on terminal size
                    adjusted_top_height = min(self.top_height, height - 4)  # Ensure at least 4 lines for bottom pane
                    
                    # Resize top pane
                    self.top_pane.resize(adjusted_top_height, width)
                    
                    # Resize and redraw divider
                    divider = curses.newwin(1, width, adjusted_top_height, 0)
                    divider.bkgd("-", curses.color_pair(5))
                    divider.refresh()
                    
                    # Resize bottom pane - ensure at least 1 line height
                    bottom_height = max(1, height - adjusted_top_height - 1)
                    self.bottom_pane.resize(bottom_height, width)
                    
                    # Update top height for future reference
                    self.top_height = adjusted_top_height
            except curses.error as e:
                pass  # Just ignore resize errors and try again next refresh
                
            # Update top pane (stats)
            try:
                self._update_stats_pane()
            except curses.error:
                pass  # Ignore errors in stats pane update
            
            # Update bottom pane (logs)
            try:
                self._update_log_pane()
            except curses.error:
                pass  # Ignore errors in log pane update
            
            # Refresh the screen
            try:
                self.screen.refresh()
            except curses.error:
                pass  # Ignore refresh errors
            
        except Exception as e:
            # Catch and handle all other errors to prevent UI crashes
            try:
                self.screen.clear()
                error_msg = f"UI Error: {str(e)[:20]}..."
                self.screen.addstr(0, 0, error_msg)
                self.screen.refresh()
            except:
                pass  # Last resort error handling
    
    def _update_stats_pane(self):
        """Update the top pane with current statistics."""
        self.top_pane.clear()
        
        # Draw header
        if self.question_type and hasattr(self.question_type, 'value') and self.question_type.value == "rt":
            generation_type = "Trace Generation"
        else:
            generation_type = "Question Generation"
        header_text = f" Argonium {generation_type} Progress - Model: {self.stats['model_name']} "
        self.top_pane.addstr(0, 0, header_text.center(self.top_pane.getmaxyx()[1], "="), 
                            curses.color_pair(5) | curses.A_BOLD)
        
        # Get width for layout calculations
        max_width = self.top_pane.getmaxyx()[1] - 4  # Leave 2 chars padding on each side
        col2_start = max(30, max_width // 2)  # Start second column at half width or 30, whichever is larger
        
        # Draw statistics
        y = 2
        # Files progress
        self.top_pane.addstr(y, 2, f"Files: ", curses.A_BOLD)
        # Make sure we don't display more files processed than total files
        displayed_processed = min(self.stats['files_processed'], self.stats['total_files'])
        file_progress = f"{displayed_processed}/{self.stats['total_files']} "
        
        if self.stats['total_files'] > 0:
            # Use the explicit file_percentage if available, otherwise calculate it
            if self.stats.get('file_percentage', 0) > 0:
                file_percentage = min(100.0, self.stats['file_percentage'])
            else:
                # Calculate percentage as fallback, ensuring it never exceeds 100%
                file_percentage = min(100.0, (displayed_processed/max(1, self.stats['total_files']))*100)
            file_progress += f"({file_percentage:.1f}%)"
        self.top_pane.addstr(file_progress)
        
        # Chunks progress - only show if there's enough width
        if max_width >= 50:  # Only show second column if screen is wide enough
            self.top_pane.addstr(y, col2_start, f"Chunks: ", curses.A_BOLD)
            # Make sure we don't display more chunks processed than total chunks
            displayed_chunks = min(self.stats['chunks_processed'], self.stats['total_chunks'])
            chunk_progress = f"{displayed_chunks}/{self.stats['total_chunks']} "
            
            if self.stats['total_chunks'] > 0:
                # Calculate percentage, ensuring it never exceeds 100%
                chunk_percentage = min(100.0, (displayed_chunks/max(1, self.stats['total_chunks']))*100)
                chunk_progress += f"({chunk_percentage:.1f}%)"
            self.top_pane.addstr(chunk_progress)
        
        # Questions/Traces generated
        y += 1
        if self.question_type and hasattr(self.question_type, 'value') and self.question_type.value == "rt":
            label = "Traces Generated: "
        else:
            label = "Questions Generated: "
        self.top_pane.addstr(y, 2, label, curses.A_BOLD)
        self.top_pane.addstr(f"{self.stats['questions_generated']}")
        
        # Success rate - only show if there's enough width
        if max_width >= 50:  # Only show second column if screen is wide enough
            self.top_pane.addstr(y, col2_start, f"Success Rate: ", curses.A_BOLD)
            success_color = curses.color_pair(1) if self.stats['success_rate'] >= 70 else curses.color_pair(3)
            self.top_pane.addstr(f"{self.stats['success_rate']:.1f}%", success_color)
        
        # Overall completion
        y += 1
        self.top_pane.addstr(y, 2, f"Overall Completion: ", curses.A_BOLD)
        # Calculate progress bar width based on available space (leave room for percentage)
        progress_width = max(10, max_width - 25)  # At least 10 chars wide, adjust for label and percentage
        
        # Ensure completion percentage never exceeds 100%
        completion_percentage = min(100.0, self.stats['completion_percentage'])
        progress_bar = self._generate_progress_bar(completion_percentage/100, progress_width)
        self.top_pane.addstr(f"{progress_bar} {completion_percentage:.1f}%")
        
        # Time information
        y += 2
        self.top_pane.addstr(y, 2, f"Elapsed Time: ", curses.A_BOLD)
        self.top_pane.addstr(f"{self._format_time(self.stats['elapsed_time'])}")
        
        # ETA - only show if there's enough width
        if max_width >= 70:  # Only show if screen is wide enough
            self.top_pane.addstr(y, col2_start, f"ETA: ", curses.A_BOLD)
            self.top_pane.addstr(f"{self.stats['eta']}")
        else:  # If screen isn't wide enough, show ETA on next line
            y += 1
            self.top_pane.addstr(y, 2, f"ETA: ", curses.A_BOLD)
            self.top_pane.addstr(f"{self.stats['eta']}")
            
        # Average chunk processing time - Add on next line
        y += 1
        self.top_pane.addstr(y, 2, f"Avg Processing Time: ", curses.A_BOLD)
        self.top_pane.addstr(f"{self.stats['avg_chunk_time']}")
        
        # Worker information
        y += 1
        self.top_pane.addstr(y, 2, f"Active Workers: ", curses.A_BOLD)
        worker_color = curses.color_pair(1) if self.stats['active_workers'] > 0 else curses.color_pair(0)
        self.top_pane.addstr(f"{self.stats['active_workers']}/{self.stats['max_workers']}", worker_color)
        
        # Current activity
        y += 2
        self.top_pane.addstr(y, 2, f"Current File: ", curses.A_BOLD)
        # Truncate if filename is too long
        max_file_len = max(10, max_width - 15)  # Adjust for label, ensure minimum width
        file_display = self.stats['current_file']
        if len(file_display) > max_file_len:
            file_display = "..." + file_display[-(max_file_len-3):]
        try:  # Use try/except to catch rendering errors when title is too long
            self.top_pane.addstr(file_display)
        except curses.error:
            pass  # Silently ignore rendering errors
        
        # Current chunk
        y += 1
        self.top_pane.addstr(y, 2, f"Current Chunk: ", curses.A_BOLD)
        # Truncate if chunk id is too long
        chunk_display = self.stats['current_chunk']
        max_chunk_len = max(10, max_width - 15)  # Adjust for label, ensure minimum width
        if len(chunk_display) > max_chunk_len:
            chunk_display = chunk_display[:max_chunk_len-3] + "..."
        try:  # Use try/except to catch rendering errors
            self.top_pane.addstr(chunk_display)
        except curses.error:
            pass  # Silently ignore rendering errors
        
        # Status message
        y += 1
        self.top_pane.addstr(y, 2, f"Status: ", curses.A_BOLD)
        status_display = self.stats['status_message']
        max_status_len = max(10, max_width - 10)  # Adjust for label, ensure minimum width
        if len(status_display) > max_status_len:
            status_display = status_display[:max_status_len-3] + "..."
        try:  # Use try/except to catch rendering errors
            self.top_pane.addstr(status_display)
        except curses.error:
            pass  # Silently ignore rendering errors
            
        # Error statistics section - Always show this section
        y += 2
        try:
            # Error header - always display it
            self.top_pane.addstr(y, 2, "ERROR STATISTICS", curses.color_pair(2) | curses.A_BOLD)
            y += 1
                
            # Total errors - always show counters
            self.top_pane.addstr(y, 2, f"Total Errors: ", curses.A_BOLD)
            error_color = curses.color_pair(2) if self.stats['total_errors'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['total_errors']}", error_color)
            
            # Low score questions (right column) - always show
            self.top_pane.addstr(y, col2_start, f"Low Scores: ", curses.A_BOLD)
            low_score_color = curses.color_pair(3) if self.stats['low_score_questions'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['low_score_questions']}", low_score_color)
            
            # Next row for detailed counters
            y += 1
            
            # File processing errors
            self.top_pane.addstr(y, 2, f"File Proc: ")
            file_color = curses.color_pair(2) if self.stats['error_file_processing'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['error_file_processing']}", file_color)
            
            # Chunk extraction errors
            self.top_pane.addstr(f"  Chunk Extract: ")
            chunk_color = curses.color_pair(2) if self.stats['error_chunk_extraction'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['error_chunk_extraction']}", chunk_color)
            
            # API errors (second column)
            self.top_pane.addstr(y, col2_start, f"API: ")
            api_color = curses.color_pair(2) if self.stats['error_api'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['error_api']}", api_color)
            
            # Next row for more detailed counters
            y += 1
            
            # Reading errors
            self.top_pane.addstr(y, 2, f"Chunk Read: ")
            read_color = curses.color_pair(2) if self.stats['error_chunk_reading'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['error_chunk_reading']}", read_color)
            
            # Summarizing errors
            self.top_pane.addstr(f"  Summary: ")
            summary_color = curses.color_pair(2) if self.stats['error_summarizing'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['error_summarizing']}", summary_color)
            
            # Question generation and evaluation
            self.top_pane.addstr(y, col2_start, f"Q-Gen: ")
            qgen_color = curses.color_pair(2) if self.stats['error_question_gen'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['error_question_gen']}", qgen_color)
            
            self.top_pane.addstr(f"  Q-Eval: ")
            qeval_color = curses.color_pair(2) if self.stats['error_question_eval'] > 0 else curses.color_pair(0)
            self.top_pane.addstr(f"{self.stats['error_question_eval']}", qeval_color)
        except curses.error:
            pass  # Safely ignore rendering errors
        
        # Footer hint
        y = self.top_height - 1
        self.top_pane.addstr(y, 2, "Press Ctrl+C to gracefully exit", curses.color_pair(3))
        
        self.top_pane.refresh()
    
    def _update_log_pane(self):
        """Update the bottom pane with log messages."""
        self.bottom_pane.clear()
        
        # Get visible area dimensions
        height, width = self.bottom_pane.getmaxyx()
        
        # Safety check for dimensions
        if height <= 0 or width <= 0:
            return  # Can't draw in a 0-sized space
        
        # Calculate which log lines to show
        start_line = max(0, self.log_position - height + 1)
        if start_line >= len(self.log_lines):
            start_line = max(0, len(self.log_lines) - 1)  # Adjust for empty log or out-of-bounds position
            
        # Create safe view into log lines
        try:
            end_index = min(len(self.log_lines), self.log_position + 1)
            visible_logs = self.log_lines[start_line:end_index]
        except Exception:
            visible_logs = ["Log error"]  # Fallback for any indexing errors
        
        # Print log lines
        for i, line in enumerate(visible_logs):
            if i >= height:
                break
                
            # Color-code certain types of messages
            color = curses.color_pair(0)  # Default
            
            # Detect message type based on content
            if "[ERROR]" in line or "error" in line.lower() or "failed" in line.lower():
                color = curses.color_pair(2)  # Red for errors
            elif "[WARNING]" in line or "warning" in line.lower():
                color = curses.color_pair(3)  # Yellow for warnings 
            elif "completed" in line.lower() or "success" in line.lower() or "generated" in line.lower():
                color = curses.color_pair(1)  # Green for success
            
            # Truncate line if it's too long
            if len(line) > width - 1:
                line = line[:width-4] + "..."
                
            try:
                self.bottom_pane.addstr(i, 0, line, color)
            except curses.error:
                pass  # Catch rendering errors
        
        # Show scroll indicator if applicable
        if len(self.log_lines) > height:
            scroll_percent = min(100, max(0, start_line * 100 // max(1, len(self.log_lines) - height)))
            scroll_indicator = f" [{scroll_percent}%] "
            
            try:
                # Place indicator at the bottom right
                self.bottom_pane.addstr(height-1, width - len(scroll_indicator), 
                                       scroll_indicator, curses.A_REVERSE)
            except curses.error:
                pass  # Ignore if there's not enough space
        
        try:
            self.bottom_pane.refresh()
        except curses.error:
            pass  # Ignore refresh errors
    
    def _generate_progress_bar(self, fraction, width):
        """Generate a text-based progress bar."""
        if width <= 2:  # Handle very small widths
            return "[]"
            
        # Ensure width is at least 3 characters ([] plus at least one character inside)
        width = max(3, width)
        
        # Calculate filled portion
        filled = int((width - 2) * fraction)  # -2 for the brackets
        empty = (width - 2) - filled  # Remaining space
        
        # Create bar with proper clipping
        return f"[{'█' * filled}{'-' * empty}]"
    
    def _format_time(self, seconds):
        """Format time in seconds to a readable string."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        else:
            return f"{seconds/86400:.1f} days"
    
    def update_stats(self, **kwargs):
        """Update the statistics in the top pane."""
        # Special handling for current_file and current_chunk - only update if provided
        # and not empty so they don't get reset when calling update_global_stats
        for key, value in kwargs.items():
            if key in self.stats:
                # For current_file and current_chunk, only update if value is non-empty
                if key in ['current_file', 'current_chunk']:
                    if value: # Only update if value is non-empty
                        self.stats[key] = value
                # For numeric stats, ensure they have valid values
                elif key in ['files_processed', 'total_files', 'chunks_processed', 'total_chunks', 
                            'chunks_extracted', 'questions_generated']:
                    # Ensure numeric values are non-negative
                    if isinstance(value, (int, float)):
                        self.stats[key] = max(0, value)
                # Special case for percentages
                elif key in ['completion_percentage', 'success_rate']:
                    # Ensure percentage values are between 0 and 100
                    if isinstance(value, (int, float)):
                        self.stats[key] = min(100.0, max(0.0, value))
                else:
                    # For all other stats, update normally
                    self.stats[key] = value
    
    def log(self, message):
        """Add a message to the log queue.
        
        Args:
            message: Either a string message or a tuple of (message, level)
        """
        if _use_split_screen:
            try:
                _log_queue.put(message, block=False)
            except queue.Full:
                # If queue is somehow full, try to log an error
                try:
                    _log_queue.put(("Log queue full - messages may be lost", "ERROR"), block=False)
                except:
                    pass  # Give up if we can't even log that
        else:
            # Fallback to normal print
            if isinstance(message, tuple) and len(message) == 2:
                msg, level = message
                prefix = "[" + level + "] " if level != "INFO" else ""
                print(f"{prefix}{msg}")
            else:
                print(message)
    
    def set_question_type(self, question_type):
        """Set the question type for display purposes."""
        self.question_type = question_type


# Create the terminal UI instance
terminal_ui = None

def init_terminal_ui():
    """Initialize the terminal UI."""
    global terminal_ui, _use_split_screen
    terminal_ui = TerminalUI()
    success = terminal_ui.start()
    
    # Register cleanup function
    atexit.register(cleanup_ui)
    
    return success

def cleanup_ui():
    """Clean up the terminal UI."""
    global terminal_ui
    if terminal_ui:
        terminal_ui.stop()
        # Wait a moment for curses to clean up
        time.sleep(0.2)

# Custom print function that redirects to UI or standard output








def batched_openai_completion(model: str, messages: list, **kwargs):
    """
    Make an OpenAI API call using standard API.
    
    Args:
        model: The model to use
        messages: The messages to send
        **kwargs: Additional parameters for the API call
        
    Returns:
        The API response as a dictionary (compatible with OpenAI API 1.0+)
    """
    global _openai_client
    
    # Prepare request parameters
    request_params = {
        "model": model,
        "messages": messages,
        **kwargs
    }
    
    try:
        response = _openai_client.chat.completions.create(**request_params)
        
        # Convert Pydantic model to dict for backward compatibility
        # This ensures existing code using response['key'] access patterns continue to work
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        else:
            # Fallback for older OpenAI library versions
            return response
            
    except Exception as e:
        # Import here to avoid circular import issues
        import openai
        
        # Detailed error logging with context
        error_details = f"OpenAI API Error: {str(e)}"
        error_context = f"Model: {model}, Messages count: {len(messages)}"
        
        # Log different types of OpenAI errors with specific details
        if hasattr(openai, 'RateLimitError') and isinstance(e, openai.RateLimitError):
            log_message(f"Rate limit exceeded. {error_details}. Context: {error_context}", 
                       log_level="ERROR", error_type="api")
        elif hasattr(openai, 'AuthenticationError') and isinstance(e, openai.AuthenticationError):
            log_message(f"Authentication failed. {error_details}. Context: {error_context}", 
                       log_level="ERROR", error_type="api")
        elif hasattr(openai, 'BadRequestError') and isinstance(e, openai.BadRequestError):
            log_message(f"Bad request sent to API. {error_details}. Context: {error_context}", 
                       log_level="ERROR", error_type="api")
        elif hasattr(openai, 'APIConnectionError') and isinstance(e, openai.APIConnectionError):
            log_message(f"Connection error to OpenAI API. {error_details}. Context: {error_context}", 
                       log_level="ERROR", error_type="api")
        elif hasattr(openai, 'APIError') and isinstance(e, openai.APIError):
            log_message(f"OpenAI API error. {error_details}. Context: {error_context}", 
                       log_level="ERROR", error_type="api")
        else:
            log_message(f"Unexpected API error. {error_details}. Context: {error_context}", 
                       log_level="ERROR", error_type="api")
        
        # Re-raise the exception so calling code can handle it appropriately
        raise


def clean_answer_content(answer_text: str) -> str:
    """
    Clean up answer content to remove evaluation commentary and keep only technical content.
    
    Args:
        answer_text: The raw answer text that may contain evaluation comments
        
    Returns:
        str: Cleaned answer text with only technical content
    """
    # Remove common evaluation phrases that should not be in technical answers
    evaluation_patterns = [
        r'The question is clear[^.]*\.',
        r'The question is well-structured[^.]*\.',
        r'The stem is unambiguous[^.]*\.',
        r'The answer choices are plausible[^.]*\.',
        r'The question is factually accurate[^.]*\.',
        r'The difficulty is appropriate[^.]*\.',
        r'The distractors are well-chosen[^.]*\.',
        r'The educational value is high[^.]*\.',
        r'The only minor deduction[^.]*\.',
        r'The content block[^.]*\.',
        r'This question tests[^.]*\.',
        r'The question requires[^.]*\.',
        r'This is a good question[^.]*\.',
        r'This question is appropriate[^.]*\.',
        r'The provided context[^.]*\.',
        r'Overall[,\s]*this is[^.]*\.',
        r'In summary[^.]*\.',
        r'The answer provided[^.]*\.',
        r'This answer[^.]*demonstrates[^.]*\.',
        r'The response shows[^.]*\.',
        r'This explanation[^.]*\.',
        r'The text states[^.]*\.',
        r'According to the passage[^.]*\.',
        r'The passage mentions[^.]*\.',
        r'As described in[^.]*\.',
        r'The document indicates[^.]*\.',
        r'This demonstrates[^.]*understanding[^.]*\.',
        r'requiring careful reading[^.]*\.',
        r'potential[ly]* overwhelm[^.]*\.',
    ]
    
    cleaned_text = answer_text
    
    # Remove evaluation patterns
    for pattern in evaluation_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove any remaining references to letter labels (A), (B), etc.
    cleaned_text = re.sub(r'\([A-Z]\)', '', cleaned_text)
    cleaned_text = re.sub(r'choice\s+[A-Z]', '', cleaned_text)
    cleaned_text = re.sub(r'option\s+[A-Z]', '', cleaned_text)
    
    # Clean up extra whitespace and normalize spacing
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s*\.\s*\.', '.', cleaned_text)  # Remove double periods
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def clean_answer_choices(choices: List[str]) -> List[str]:
    """
    Clean up individual answer choices to remove evaluation content.
    
    Args:
        choices: List of answer choice strings
        
    Returns:
        List of cleaned answer choice strings
    """
    cleaned_choices = []
    for choice in choices:
        # Remove any evaluation commentary from individual choices
        cleaned_choice = clean_answer_content(choice)
        cleaned_choices.append(cleaned_choice)
    
    return cleaned_choices


def log_message(message, log_level="INFO", error_type=None):
    """
    Log a message to the appropriate output and track errors by category.
    
    Args:
        message: The message to log
        log_level: The severity level of the message (INFO, WARNING, ERROR, DEBUG)
        error_type: Optional type of error for categorization and counting
                   One of: file_processing, chunk_extraction, chunk_reading, 
                           summarizing, question_gen, question_eval, api, other, low_score
    """
    global terminal_ui, _use_split_screen, _error_log_file, _exit_requested
    
    # For critical log messages that indicate file or chunk count changes, 
    # make sure UI is kept in sync by forcing an update_global_stats call
    force_ui_update = False
    if "processed" in message and "files" in message and ("/" in message or "%" in message):
        force_ui_update = True
    
    # Track errors by category if this is an error message
    if log_level == "ERROR" or error_type == "low_score":
        # Get checkpoint manager if available
        checkpoint_manager = CheckpointManager.get_instance()
        current_error_count = 0
        error_type_str = error_type if error_type else "general"
        
        # Log error to the error log file if it's initialized
        if _error_log_file:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(_error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] [{log_level}] [{error_type_str}] {message}\n")
            except Exception as e:
                # Don't let error logging errors crash the program
                print(f"Failed to write to error log: {str(e)}")
        
        # Update error counters in the terminal UI
        if _use_split_screen and terminal_ui:
            # Increment total errors counter
            terminal_ui.stats["total_errors"] += 1
            current_error_count = terminal_ui.stats["total_errors"]
            
            # Categorize and count the error
            if error_type == "file_processing":
                terminal_ui.stats["error_file_processing"] += 1
                current_error_count = terminal_ui.stats["error_file_processing"]
            elif error_type == "chunk_extraction":
                terminal_ui.stats["error_chunk_extraction"] += 1
                current_error_count = terminal_ui.stats["error_chunk_extraction"]
            elif error_type == "chunk_reading":
                terminal_ui.stats["error_chunk_reading"] += 1
                current_error_count = terminal_ui.stats["error_chunk_reading"]
            elif error_type == "summarizing":
                terminal_ui.stats["error_summarizing"] += 1
                current_error_count = terminal_ui.stats["error_summarizing"]
            elif error_type == "question_gen":
                terminal_ui.stats["error_question_gen"] += 1
                current_error_count = terminal_ui.stats["error_question_gen"]
            elif error_type == "question_eval":
                terminal_ui.stats["error_question_eval"] += 1
                current_error_count = terminal_ui.stats["error_question_eval"]
            elif error_type == "api":
                terminal_ui.stats["error_api"] += 1
                current_error_count = terminal_ui.stats["error_api"]
            elif error_type == "low_score":
                terminal_ui.stats["low_score_questions"] += 1
                current_error_count = terminal_ui.stats["low_score_questions"]
            else:
                terminal_ui.stats["error_other"] += 1
                current_error_count = terminal_ui.stats["error_other"]
        
        # Also update error stats in checkpoint if available
        if checkpoint_manager:
            # Increment total errors counter
            checkpoint_manager.checkpoint_data['error_stats']['total_errors'] += 1
            
            # Categorize and count the error
            if error_type == "file_processing":
                checkpoint_manager.checkpoint_data['error_stats']['error_file_processing'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_file_processing'])
            elif error_type == "chunk_extraction":
                checkpoint_manager.checkpoint_data['error_stats']['error_chunk_extraction'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_chunk_extraction'])
            elif error_type == "chunk_reading":
                checkpoint_manager.checkpoint_data['error_stats']['error_chunk_reading'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_chunk_reading'])
            elif error_type == "summarizing":
                checkpoint_manager.checkpoint_data['error_stats']['error_summarizing'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_summarizing'])
            elif error_type == "question_gen":
                checkpoint_manager.checkpoint_data['error_stats']['error_question_gen'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_question_gen'])
            elif error_type == "question_eval":
                checkpoint_manager.checkpoint_data['error_stats']['error_question_eval'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_question_eval'])
            elif error_type == "api":
                checkpoint_manager.checkpoint_data['error_stats']['error_api'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_api'])
            elif error_type == "low_score":
                checkpoint_manager.checkpoint_data['error_stats']['low_score_questions'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['low_score_questions'])
            else:
                checkpoint_manager.checkpoint_data['error_stats']['error_other'] += 1
                current_error_count = max(current_error_count, checkpoint_manager.checkpoint_data['error_stats']['error_other'])
                
            # Save checkpoint periodically (not on every error to reduce I/O)
            if time.time() - checkpoint_manager.last_save_time > 30:  # Save at most every 30 seconds
                checkpoint_manager.save()
        
        # Check if we've exceeded the error threshold
        if current_error_count >= _max_error_threshold:
            error_msg = f"Error threshold exceeded ({current_error_count} errors of type '{error_type_str}'). Stopping process."
            # Log the threshold exceeded error to the log file
            if _error_log_file:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(_error_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{timestamp}] [CRITICAL] {error_msg}\n")
                except Exception:
                    pass  # If we can't log, continue with shutdown
            
            # Set exit flag
            _exit_requested = True
            
            # Show critical message
            print(f"\n\nCRITICAL: {error_msg}\n\n")
            
            # Update UI if available
            if _use_split_screen and terminal_ui:
                terminal_ui.update_stats(
                    status_message=f"STOPPED: {error_msg}"
                )
    
    # Forward to UI if applicable
    if _use_split_screen and terminal_ui:
        # Send message and level to the UI
        terminal_ui.log((message, log_level))
        
        # For critical log messages about file/chunk processing, force a UI update
        # to ensure the display is in sync with the log message
        if force_ui_update:
            update_global_stats()
    else:
        # Fallback to standard print with optional prefix
        prefix = ""
        if log_level == "ERROR":
            prefix = "[ERROR] "
        elif log_level == "WARNING":
            prefix = "[WARNING] "
        elif log_level == "DEBUG" and os.environ.get("DEBUG"):
            prefix = "[DEBUG] "
        elif log_level == "DEBUG":
            # Skip debug messages unless DEBUG env var is set
            return
        print(f"{prefix}{message}")

# Function to update global statistics in the UI
def update_global_stats():
    """Update global statistics in the terminal UI."""
    global terminal_ui, _use_split_screen
    global _total_files, _processed_files, _total_chunks, _processed_chunks, _extracted_chunks
    global _active_workers, _max_workers
    
    if not _use_split_screen or not terminal_ui:
        return
        
    # Get checkpoint manager for additional stats
    checkpoint_manager = CheckpointManager.get_instance()
    questions_generated = 0
    completion_percentage = 0
    total_completed_chunks = 0
    
    # Calculate file percentage based on the latest counts
    file_percentage = min(100.0, (_processed_files / max(1, _total_files)) * 100)
    
    if checkpoint_manager:
        questions_generated = checkpoint_manager.get_questions_count()
        stats = checkpoint_manager.get_completion_stats()
        # Ensure completion percentage doesn't exceed 100%
        completion_percentage = min(100.0, stats.get('completion_percentage', 0))
        total_completed_chunks = stats.get('completed_chunks', 0)
    
    # Calculate success rate using both checkpoint data and current session data
    success_rate = 0
    
    # Get completed chunks (with any status) from checkpoint
    if checkpoint_manager:
        # Count chunks with 'completed' status (successful question generation)
        success_count = sum(1 for c in checkpoint_manager.get_processed_chunks().values() 
                         if c.get('status') == 'completed')
        
        # Only count chunks with an actual completion status (completed, low_score, error)
        processed_count = sum(1 for c in checkpoint_manager.get_processed_chunks().values() 
                           if c.get('status') in ['completed', 'low_score', 'error'])
        
        # Calculate success rate: successful chunks / all processed chunks
        if processed_count > 0:
            success_rate = (success_count / processed_count) * 100
    
    # Calculate elapsed time and ETA
    elapsed_time = time.time() - _start_time
    
    # Track processing times for better ETA calculation
    if not hasattr(update_global_stats, 'last_processed_count'):
        update_global_stats.last_processed_count = 0
        update_global_stats.last_time_check = time.time()
        update_global_stats.chunk_processing_times = []
    
    # Check if new chunks have been processed since last call
    current_processed = _processed_chunks
    if current_processed > update_global_stats.last_processed_count:
        # Calculate time per chunk for this interval
        chunks_processed = current_processed - update_global_stats.last_processed_count
        time_passed = time.time() - update_global_stats.last_time_check
        
        if chunks_processed > 0 and time_passed > 0:
            # Calculate time per chunk and add to our history
            time_per_chunk = time_passed / chunks_processed
            update_global_stats.chunk_processing_times.append(time_per_chunk)
            
            # Keep only the most recent 10 measurements
            if len(update_global_stats.chunk_processing_times) > 10:
                update_global_stats.chunk_processing_times = update_global_stats.chunk_processing_times[-10:]
    
    # Update tracking variables
    update_global_stats.last_processed_count = current_processed
    update_global_stats.last_time_check = time.time()
    
    # Estimate remaining time based on completion percentage and recent processing rates
    eta = "Unknown"
    avg_chunk_time = None
    avg_time_str = "Unknown"
    
    # Only estimate if we have processed at least a few chunks for a reasonable estimate
    min_processed = 5  # Require at least 5 chunks processed
    if completion_percentage > 0 and elapsed_time > 30 and len(update_global_stats.chunk_processing_times) >= 3:
        # Use weighted average favoring recent times
        total_weight = 0
        weighted_sum = 0
        
        # Assign increasing weights to more recent measurements
        for i, time_value in enumerate(update_global_stats.chunk_processing_times):
            # Weight increases with recency
            weight = i + 1
            weighted_sum += time_value * weight
            total_weight += weight
        
        # Calculate weighted average time per chunk
        if total_weight > 0:
            avg_chunk_time = weighted_sum / total_weight
            
            # Format average time per chunk in a readable format
            if avg_chunk_time < 1:
                avg_time_str = f"{avg_chunk_time * 1000:.0f} ms/chunk"
            elif avg_chunk_time < 60:
                avg_time_str = f"{avg_chunk_time:.1f} sec/chunk"
            else:
                avg_time_str = f"{avg_chunk_time / 60:.1f} min/chunk"
                
            # Calculate completion percentage (with safety checks)
            if _total_chunks == 0:
                completion_percentage = 100.0  # If there are no chunks, we're done
            else:
                completion_percentage = min(100.0, max(0.1, (total_completed_chunks / _total_chunks) * 100))
            
            # If we've completed all chunks, ETA is 0
            if completion_percentage >= 100:
                eta = "0 seconds"
            else:
                # Simple and robust ETA calculation based on percentage completion
                # If we've done X% in Y seconds, then remaining (100-X)% will take Y * (100-X)/X seconds
                
                # Calculate time per percentage point (seconds per %)
                time_per_percentage = elapsed_time / completion_percentage
                
                # Calculate remaining percentage
                remaining_percentage = 100.0 - completion_percentage
                
                # Calculate remaining time based on current rate
                remaining_time = time_per_percentage * remaining_percentage
                
                # Ensure remaining time is not negative
                remaining_time = max(0, remaining_time)
                
                # Format the ETA string using human_readable_time function
                eta = human_readable_time(remaining_time)
    
    # Use existing values for current file and chunk (don't reset them)
    status_message = "Processing..."
    
    # Get the latest counter values in a thread-safe way
    with _counter_lock:
        current_processed_files = _processed_files
        current_total_files = _total_files
        current_extracted_chunks = _extracted_chunks
        current_total_chunks = _total_chunks
        current_processed_chunks = _processed_chunks
    
    # Always recalculate file percentage based on the latest counter values
    file_percentage = min(100.0, (current_processed_files / max(1, current_total_files)) * 100)
    
    # Only update stats we've calculated, not overriding current file/chunk
    terminal_ui.update_stats(
        # File statistics - these are critical for displaying correct progress
        files_processed=current_processed_files, 
        total_files=current_total_files,
        
        # Chunk statistics
        chunks_extracted=current_extracted_chunks,
        chunks_processed=current_processed_chunks,
        total_chunks=current_total_chunks,
        
        # Progress statistics
        completion_percentage=completion_percentage,
        file_percentage=file_percentage,  # Pass the file percentage explicitly
        
        # Other statistics
        questions_generated=questions_generated,
        success_rate=success_rate,
        elapsed_time=elapsed_time,
        eta=eta,
        avg_chunk_time=avg_time_str,
        status_message=status_message,
        active_workers=_active_workers,
        max_workers=_max_workers
    )

##############################################################################
# Worker Pool Management for Parallel Processing
##############################################################################

def init_worker_pool(max_workers=None):
    """
    Initialize the worker pool for parallel processing.
    
    Args:
        max_workers: Maximum number of worker processes to use
                    (defaults to None, which will use CPU count)
    """
    global _worker_pool, _max_workers, _result_queue
    
    # Set max workers based on CPU count if not specified
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free for the main process
    
    _max_workers = max_workers
    log_message(f"Initializing worker pool with {_max_workers} workers")
    
    # Create result queue for workers to return their results
    _result_queue = queue.Queue()
    
    # Create the worker pool - use ThreadPoolExecutor for better OpenAI API connection sharing
    # Use ProcessPoolExecutor for CPU-bound tasks
    _worker_pool = ThreadPoolExecutor(max_workers=max_workers)
    
    log_message(f"Worker pool initialized successfully")
    return _worker_pool

def shutdown_worker_pool():
    """
    Gracefully shut down the worker pool.
    This function cancels pending futures and waits for running tasks to complete.
    """
    global _worker_pool, _active_workers, _exit_requested
    
    if _worker_pool:
        log_message("Shutting down worker pool...", log_level="WARNING")
        
        # Ensure exit flag is set
        _exit_requested = True
        
        # Try to cancel any pending futures - this needs to be done manually
        # because ThreadPoolExecutor.shutdown() doesn't automatically cancel futures
        try:
            # Get access to internal futures queue (this is a bit of a hack)
            for future in list(_worker_pool._work_queue.queue):
                if not future.done() and not future.running():
                    future.cancel()
            log_message("Cancelled pending tasks", log_level="WARNING")
        except (AttributeError, Exception) as e:
            log_message(f"Could not cancel pending tasks: {e}", log_level="WARNING")
        
        # Shutdown with wait=True to allow running tasks to complete
        # Use a timeout to prevent hanging indefinitely
        try:
            # First attempt a clean shutdown with a timeout
            _worker_pool.shutdown(wait=False)
            log_message("Initiated worker pool shutdown", log_level="WARNING")
        except Exception as e:
            log_message(f"Error during worker pool shutdown: {e}", log_level="ERROR")
        
        # Reset worker count
        _active_workers = 0
        log_message("Worker pool shutdown complete", log_level="WARNING")

def worker_process_chunk(chunk_id, chunk_text, model_name, question_type, num_answers, min_score):
    """
    Worker function to process a single chunk in a separate process.
    This function handles only the question generation part, not IO.
    
    Args:
        chunk_id: The unique ID of the chunk
        chunk_text: The text content of the chunk
        model_name: The name of the LLM to use
        question_type: Type of question to generate
        num_answers: Number of answers for multiple-choice questions
        min_score: Minimum score threshold for keeping a question
        
    Returns:
        Dictionary with the generated question/answer pair and metadata
    """
    try:
        # Update global active worker count
        global _active_workers, _exit_requested
        with _counter_lock:
            _active_workers += 1
        
        # Check for shutdown signal before starting work
        if _exit_requested:
            with _counter_lock:
                _active_workers -= 1
            return {
                'chunk_id': chunk_id,
                'status': 'cancelled',
                'processing_time': 0,
                'message': 'Cancelled due to shutdown request'
            }
        
        # Process based on question type
        if question_type == QuestionType.MULTIPLE_CHOICE:
            result = generate_multiple_choice_qa_pairs(
                chunk_id, chunk_text, model_name, num_answers, min_score
            )
        elif question_type == QuestionType.FREE_FORM:
            result = generate_free_form_qa_pairs(
                chunk_id, chunk_text, model_name, min_score
            )
        else:  # REASONING_TRACE
            result = generate_reasoning_trace_pairs(
                chunk_id, chunk_text, model_name, min_score
            )
        
        # Clean up and return result
        with _counter_lock:
            _active_workers -= 1
        
        # Check again for shutdown signal before returning result
        if _exit_requested:
            return {
                'chunk_id': chunk_id,
                'status': 'cancelled',
                'processing_time': 0,
                'message': 'Cancelled due to shutdown request'
            }
        
        return result
    
    except Exception as e:
        # Handle any unexpected errors
        error_message = f"Worker error processing chunk {chunk_id}: {str(e)}"
        
        # Clean up in case of error
        with _counter_lock:
            _active_workers -= 1
        
        # Return error result
        return {
            'chunk_id': chunk_id,
            'error': error_message,
            'status': 'error',
            'processing_time': 0
        }

def submit_chunk_to_worker_pool(chunk_id, chunk_text, model_name, question_type, num_answers, min_score):
    """
    Submit a chunk to the worker pool for processing.
    
    Args:
        chunk_id: The unique ID of the chunk
        chunk_text: The text content of the chunk
        model_name: The name of the model to use
        question_type: Type of question to generate
        num_answers: Number of answers for multiple-choice questions
        min_score: Minimum score threshold for keeping a question
        
    Returns:
        Future object representing the pending result
    """
    global _worker_pool
    
    if _worker_pool is None:
        raise ValueError("Worker pool not initialized. Call init_worker_pool() first.")
    
    # Submit the task to the worker pool
    future = _worker_pool.submit(
        worker_process_chunk,
        chunk_id, chunk_text, model_name, question_type, num_answers, min_score
    )
    
    return future

##############################################################################
# Constants and Enums
##############################################################################
class QuestionType(Enum):
    MULTIPLE_CHOICE = "mc"
    FREE_FORM = "qa"
    REASONING_TRACE = "rt"
    
    def __str__(self):
        return self.value


##############################################################################
# Utility functions
##############################################################################

def ensure_dir_exists(dir_path: str) -> None:
    """
    Ensure that the directory exists, create if it doesn't.
    """
    os.makedirs(dir_path, exist_ok=True)


def human_readable_time(seconds: float) -> str:
    """
    Convert time in seconds into a more human-friendly format
    (seconds, minutes, hours, days).
    """
    # Less than a minute
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    # Less than an hour
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    # Less than a day
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    # More than a day
    else:
        days = seconds / 86400
        return f"{days:.2f} days"


def signal_handler(signum, frame):
    """
    Signal handler for graceful shutdown on SIGINT and SIGTERM.
    """
    global _exit_requested, terminal_ui, _worker_pool
    
    # Set the exit flag to notify main process to stop
    _exit_requested = True
    
    # Log shutdown message
    shutdown_msg = "Interrupt received. Beginning graceful shutdown process..."
    log_message("\n" + "=" * 70, log_level="WARNING")
    log_message(shutdown_msg, log_level="WARNING")
    log_message("=" * 70, log_level="WARNING")
    
    # Update UI status if available
    if terminal_ui and _use_split_screen:
        terminal_ui.update_stats(status_message="Shutting down... please wait")
    
    # Gracefully shutdown the worker pool if active
    if _worker_pool:
        log_message("Shutting down worker pool...", log_level="WARNING")
        shutdown_worker_pool()
    
    # Sleep briefly to allow the message to be displayed
    time.sleep(0.5)


def generate_file_id(file_path: str) -> str:
    """
    Generate a unique identifier for a file using its path and last modified time.
    """
    try:
        mod_time = os.path.getmtime(file_path)
        file_info = f"{file_path}_{mod_time}"
        # Use SHA-256 for generating a unique ID
        file_id = hashlib.sha256(file_info.encode()).hexdigest()[:16]
        return file_id
    except Exception as e:
        log_message(f"Error generating file ID for {file_path}: {e}", log_level="ERROR", error_type="file_processing")
        basename = os.path.basename(file_path)
        # Fallback to a simpler ID if needed
        return f"file_{basename}_{int(time.time())}"


def find_files_recursively(directory: str, extensions: list) -> list:
    """
    Recursively find all files with given extensions in directory and its subdirectories.
    
    Args:
        directory: The root directory to search
        extensions: List of file extensions to include (e.g., ['.pdf', '.txt', '.md', '.mmd'])
        
    Returns:
        List of tuples (relative_path, absolute_path)
    """
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                abs_path = os.path.join(root, file)
                result.append((rel_path, abs_path))
    return result


def extract_and_write_chunks(file_id: str, file_info: Dict, chunks_dir: str,
                             chunk_size: int, checkpoint_manager = None) -> List[str]:
    """
    Extract text from a file, split into chunks, and write each chunk to a file.
    Returns a list of chunk IDs.
    """
    global _extracted_chunks, _exit_requested
    
    # Check for exit request first
    if _exit_requested:
        return []
    
    chunk_ids = []
    file_path = file_info['file_path']
    file_type = file_info['type']
    
    try:
        # Extract text based on file type
        if file_type == 'pdf':
            text = extract_text_from_pdf(file_path)
        else:  # txt or md
            text = extract_text_from_txt(file_path)
        
        if not text:
            log_message(f"Warning: No text extracted from {file_path}", log_level="WARNING", error_type="file_processing")
            return []
        
        # Split text into chunks
        chunks = split_text_into_chunks(text, chunk_size)
        
        if not chunks:
            log_message(f"Warning: No chunks created from {file_path}", log_level="WARNING", error_type="chunk_extraction")
            return []
        
        # Write each chunk to a file
        for i, chunk_text in enumerate(chunks):
            # Check for exit request
            if _exit_requested:
                log_message(f"Interrupt detected. Stopping chunk processing for {file_path}.", log_level="WARNING")
                break
                
            chunk_id = create_chunk_id(file_id, i)
            chunk_file_path = write_chunk_to_file(chunk_id, chunk_text, chunks_dir)
            
            if chunk_file_path:
                chunk_ids.append(chunk_id)
                
                # Update global chunk map
                _chunk_map[chunk_id] = {
                    'file_id': file_id,
                    'chunk_index': i,
                    'file_path': chunk_file_path,
                    'status': 'extracted',
                    'extraction_time': time.time()
                }
                
                # Update global counter in thread-safe way
                with _counter_lock:
                    _extracted_chunks += 1

                # Save to checkpoint immediately if checkpoint manager is provided
                if checkpoint_manager:
                    chunk_data = {
                        'file_id': file_id,
                        'chunk_index': i,
                        'file_path': chunk_file_path,
                        'status': 'extracted',
                        'extraction_time': time.time()
                    }
                    checkpoint_manager.update_processed_chunk(chunk_id, chunk_data)
        
        log_message(f"Extracted {len(chunks)} chunks from {file_path}")
        
        return chunk_ids
        
    except Exception as e:
        log_message(f"Error processing {file_path}: {e}", log_level="ERROR", error_type="file_processing")
        return []


def extract_chunks_sequentially(file_map: Dict[str, Dict], chunks_dir: str,
                               chunk_size: int, checkpoint_manager = None) -> Dict[str, List[str]]:
    """
    Extract chunks from multiple files sequentially.
    Returns a mapping of file IDs to lists of chunk IDs.
    """
    global _total_files, _processed_files, _total_chunks, terminal_ui, _exit_requested
    
    # Set total files counter
    _total_files = len(file_map)
    
    # Create the chunks directory if it doesn't exist
    ensure_dir_exists(chunks_dir)
    
    # Map of file IDs to lists of chunk IDs
    file_to_chunks = {}
    
    # Setup progress bar for non-UI mode only
    if not _use_split_screen or not terminal_ui:
        progress_bar = tqdm(file_map.items(), desc="Extracting chunks", unit="file")
    else:
        # For UI mode, we'll directly iterate
        progress_bar = file_map.items()
        
    # Process files one by one
    for file_id, file_info in progress_bar:
        # Check for exit request
        if _exit_requested:
            log_message("Interrupt detected. Stopping chunk extraction.", log_level="WARNING")
            break
        try:
            filename = file_info.get('filename', 'unknown')
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(
                    current_file=filename,
                    status_message=f"Extracting chunks from {filename}"
                )
                update_global_stats()
                
            log_message(f"Extracting chunks from {filename}")
            
            chunk_ids = extract_and_write_chunks(file_id, file_info, chunks_dir, 
                                              chunk_size, checkpoint_manager)
            file_to_chunks[file_id] = chunk_ids
            
            # Update global counters in a thread-safe way
            with _counter_lock:
                _processed_files += 1
                _total_chunks += len(chunk_ids)
            
            # Update file status
            if file_id in _file_map:
                _file_map[file_id]['status'] = 'chunked'
                _file_map[file_id]['chunks_count'] = len(chunk_ids)
                _file_map[file_id]['chunking_time'] = time.time()
                
                # Update the checkpoint with the processed file
                if checkpoint_manager:
                    checkpoint_manager.add_processed_file(file_id, _file_map[file_id], chunk_ids)
                
            # Update UI with all relevant stats
            if terminal_ui and _use_split_screen:
                # Calculate progress percentage for UI
                file_progress = min(100.0, (_processed_files / max(1, _total_files)) * 100)
                
                terminal_ui.update_stats(
                    files_processed=_processed_files,
                    total_chunks=_total_chunks,
                    chunks_extracted=_extracted_chunks,
                    status_message=f"Extracted {len(chunk_ids)} chunks from {filename}",
                    completion_percentage=file_progress
                )
                update_global_stats()
                
        except Exception as e:
            log_message(f"Error extracting chunks from {file_id}: {e}", log_level="ERROR", error_type="chunk_extraction")
            
            # Update file status on error
            if file_id in _file_map:
                _file_map[file_id]['status'] = 'error'
                _file_map[file_id]['error'] = str(e)
    
    return file_to_chunks
    

def check_content_relevance(chunk_text: str, model_name: str) -> Dict:
    """
    Check if the chunk content is relevant to the paper's core content.
    Returns relevance score and reasoning.
    """
    system_message = (
        "You are an expert content evaluator who determines if text content is relevant "
        "to the core scientific/technical content of a paper versus non-relevant material "
        "like copyright notices, licensing information, references, acknowledgments, or metadata."
    )
    
    user_message = (
        f"Evaluate the following text chunk and determine if it contains core scientific/technical content "
        f"that would be appropriate for generating educational questions.\n\n"
        f"TEXT CHUNK:\n{chunk_text}\n\n"
        f"EVALUATION CRITERIA:\n"
        f"- CORE CONTENT (High relevance): Scientific concepts, research findings, technical explanations, "
        f"methodology, data analysis, theories, experimental results, clinical information, etc.\n"
        f"- NON-CORE CONTENT (Low relevance): Copyright notices, licensing text, reference lists, "
        f"acknowledgments, author information, publication metadata, figure/table captions only, "
        f"page headers/footers, disclaimers, etc.\n\n"
        f"SCORING:\n"
        f"- Score 8-10: Rich core content ideal for question generation\n"
        f"- Score 5-7: Some core content but mixed with non-relevant material\n"
        f"- Score 1-4: Primarily non-relevant content (references, metadata, etc.)\n\n"
        f"Provide your response in this format:\n"
        f"RELEVANCE_SCORE: <numeric score between 1-10>\n"
        f"REASONING: <brief explanation of why this content is or isn't relevant for question generation>\n"
        f"CONTENT_TYPE: <primary type of content: 'core_scientific', 'mixed', 'references', 'metadata', 'copyright', etc.>\n"
    )
    
    try:
        response = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        
        output = response['choices'][0]['message']['content'].strip()
        
        # Extract relevance score
        score_match = re.search(r"RELEVANCE_SCORE:\s*(\d+(?:\.\d+)?)", output)
        relevance_score = int(float(score_match.group(1))) if score_match else 5
        
        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\s*(.*?)(?:\n|$)", output, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        # Extract content type
        content_type_match = re.search(r"CONTENT_TYPE:\s*(.*?)(?:\n|$)", output)
        content_type = content_type_match.group(1).strip() if content_type_match else "unknown"
        
        return {
            'relevance_score': relevance_score,
            'reasoning': reasoning,
            'content_type': content_type,
            'is_relevant': relevance_score >= 6,  # Threshold for relevance
            'raw_output': output
        }
        
    except Exception as e:
        log_message(f"Error checking content relevance: {e}", log_level="ERROR", error_type="relevance_check")
        return {
            'relevance_score': 5,  # Default to medium relevance on error
            'reasoning': f"Error during relevance check: {str(e)}",
            'content_type': 'unknown',
            'is_relevant': True,  # Default to relevant on error to avoid losing content
            'raw_output': ""
        }


def process_chunk(chunk_id: str, chunks_dir: str, model_name: str, 
                 question_type: QuestionType, num_answers: int, min_score: int, 
                 checkpoint_manager):
    """
    Process a single chunk to generate a question-answer pair.
    """
    global _processed_chunks, _exit_requested, terminal_ui
    
    # Check for exit request
    if _exit_requested:
        return None
    
    # Skip if this chunk has already been processed
    if checkpoint_manager and checkpoint_manager.is_chunk_processed(chunk_id):
        log_message(f"Chunk {chunk_id} already processed, skipping")
        return None
    
    # Read the chunk file
    chunk_subdir = chunk_id[:2]
    chunk_file_path = os.path.join(chunks_dir, chunk_subdir, f"{chunk_id}.txt")
    
    try:
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            chunk_text = f.read()
    except FileNotFoundError as e:
        log_message(f"Chunk file not found: {chunk_file_path}. Chunk ID: {chunk_id}. Error: {e}", 
                   log_level="ERROR", error_type="chunk_reading")
    except PermissionError as e:
        log_message(f"Permission denied reading chunk file: {chunk_file_path}. Chunk ID: {chunk_id}. Error: {e}", 
                   log_level="ERROR", error_type="chunk_reading")
    except UnicodeDecodeError as e:
        log_message(f"Unicode decode error reading chunk file: {chunk_file_path}. Chunk ID: {chunk_id}. Error: {e}", 
                   log_level="ERROR", error_type="chunk_reading")
    except OSError as e:
        log_message(f"OS error reading chunk file: {chunk_file_path}. Chunk ID: {chunk_id}. Error: {e}", 
                   log_level="ERROR", error_type="chunk_reading")
    except Exception as e:
        log_message(f"Unexpected error reading chunk {chunk_id} from {chunk_file_path}: {e}", 
                   log_level="ERROR", error_type="chunk_reading")
        
        # Update chunk status on error
        if checkpoint_manager:
            checkpoint_manager.update_processed_chunk(chunk_id, {
                'status': 'error',
                'error': f"Error reading chunk: {str(e)}",
                'error_time': time.time()
            })
        
        return None
    
    # Update global map to show this chunk is being processed
    if chunk_id in _chunk_map:
        _chunk_map[chunk_id]['status'] = 'processing'
        _chunk_map[chunk_id]['processing_start'] = time.time()
    
    # Update UI with current processing information
    if terminal_ui and _use_split_screen:
        status_msg = f"Processing chunk {chunk_id} - "
        if question_type == QuestionType.MULTIPLE_CHOICE:
            status_msg += "Multiple Choice"
        elif question_type == QuestionType.FREE_FORM:
            status_msg += "Free Form"
        else:  # REASONING_TRACE
            status_msg += "Reasoning Trace"
        terminal_ui.update_stats(
            current_chunk=chunk_id,
            status_message=status_msg
        )
    
    # Process the chunk based on question type
    if question_type == QuestionType.MULTIPLE_CHOICE:
        log_message(f"Generating multiple-choice question for chunk {chunk_id}")
        result = generate_multiple_choice_qa_pairs(
            chunk_id, chunk_text, model_name, num_answers, min_score
        )
    elif question_type == QuestionType.FREE_FORM:
        log_message(f"Generating free-form question for chunk {chunk_id}")
        result = generate_free_form_qa_pairs(
            chunk_id, chunk_text, model_name, min_score
        )
    else:  # REASONING_TRACE
        log_message(f"Generating reasoning trace for chunk {chunk_id}")
        result = generate_reasoning_trace_pairs(
            chunk_id, chunk_text, model_name, min_score
        )
    
    # Update global counter
    with _counter_lock:
        _processed_chunks += 1
    
    # Update chunk status in the checkpoint
    if checkpoint_manager and result:
        checkpoint_manager.update_processed_chunk(chunk_id, result)
    
    # If this chunk was in our global map, update its status
    if chunk_id in _chunk_map:
        _chunk_map[chunk_id]['status'] = result.get('status', 'error')
        _chunk_map[chunk_id]['processing_end'] = time.time()
    
    return result


def generate_multiple_choice_qa_pairs(chunk_id: str, chunk_text: str, model_name: str, 
                                      num_answers: int = 7, min_score: int = 7) -> Dict:
    """
    Generate a multiple-choice Q/A pair from a chunk.
    Returns a dictionary with question, answer, and other metadata.
    """
    global _exit_requested
    
    # Start timing the processing
    start_time = time.time()
    
    # Check for exit request
    if _exit_requested:
        return {
            'chunk_id': chunk_id,
            'status': 'cancelled',
            'processing_time': 0,
            'message': 'Cancelled due to shutdown request'
        }
    
    # --------------------------------------------------------------------
    # Step 0: Check content relevance
    # --------------------------------------------------------------------
    relevance_check = check_content_relevance(chunk_text, model_name)
    
    # Skip non-relevant content
    if not relevance_check['is_relevant']:
        log_message(f"Chunk {chunk_id} skipped - not relevant to core content: {relevance_check['reasoning']}", 
                   log_level="INFO", error_type="content_filter")
        return {
            'chunk_id': chunk_id,
            'status': 'filtered_non_relevant',
            'processing_time': time.time() - start_time,
            'relevance_check': relevance_check,
            'message': f"Skipped non-relevant content: {relevance_check['content_type']}"
        }
    
    # --------------------------------------------------------------------
    # Step 1: Summarize & expand the chunk => augmented_chunk
    # --------------------------------------------------------------------
    system_message = (
        "You are a helpful assistant that summarizes text in bullet points "
        "and expands on them using your broader knowledge. "
        "Name this result 'augmented_chunk'."
    )
    user_message = (
        f"Given the following chunk of text, please:\n\n"
        f"1. Summarize the text in bullet points.\n"
        f"2. Expand on the summary using your parametric knowledge.\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        f"Return the result as plain text labeled 'augmented_chunk:' at the start."
    )

    try:
        step1_start_time = time.time()
        response_1 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )
        step1_output = response_1['choices'][0]['message']['content'].strip()
        step1_time = time.time() - step1_start_time
        
        # We'll assume the model starts with "augmented_chunk:"
        augmented_chunk = step1_output
        if "augmented_chunk:" in step1_output.lower():
            augmented_chunk = re.split(
                r'augmented_chunk\s*:\s*',
                step1_output,
                flags=re.IGNORECASE,
                maxsplit=1
            )[-1].strip()

    except Exception as e:
        log_message(f"Error summarizing chunk {chunk_id}: {e}", log_level="ERROR", error_type="summarizing")
        return {
            'chunk_id': chunk_id,
            'error': f"Error in step 1: {str(e)}",
            'status': 'error',
            'processing_time': time.time() - start_time
        }

    # --------------------------------------------------------------------
    # Step 2: Generate a MULTIPLE-CHOICE question with n answers
    # --------------------------------------------------------------------
    # Randomly select which position should be the correct answer to ensure uniform distribution
    target_correct_position = random.randint(1, num_answers)
    
    system_message_2 = (
        "You are a helpful assistant that generates high-quality multiple-choice questions "
        "based on text provided by the user. Each question should be challenging but fair, "
        "with one clearly correct answer and plausible but incorrect distractors."
    )
    
    user_message_2 = (
        f"Generate ONE well-formed multiple-choice question "
        f"with exactly {num_answers} answer choices labeled 1 through {num_answers}.\n\n"
        f"Text:\n{augmented_chunk}\n\n"
        f"Requirements:\n"
        f"1. Begin with 1-2 sentences of contextual information that establishes the domain/topic without referencing source materials.\n"
        f"2. Create a challenging question that tests deep understanding.\n"
        f"3. Ensure there is EXACTLY ONE clearly correct answer.\n"
        f"4. Make the other choices plausible but clearly incorrect.\n"
        f"5. The question should focus on a concept or fact that is clearly stated or strongly implied in the text.\n"
        f"6. Number your answer choices from 1 to {num_answers}.\n"
        f"7. IMPORTANT: Place the correct answer in position {target_correct_position}. The correct answer must be choice number {target_correct_position}.\n"
        f"8. DO NOT provide explanations for why each answer is correct or incorrect.\n"
        f"9. CRITICAL: Both context and question must be completely self-contained. DO NOT reference any external materials including:\n"
        f"   - 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"   - 'the author states', 'according to the text', 'as mentioned', 'as described'\n"
        f"   - 'Appendix', 'Figure', 'Table', 'Section', 'Chapter', 'above', 'below'\n"
        f"   - Any other references to source materials or external content\n"
        f"10. The context and question should read as if testing general knowledge on the topic, not comprehension of a specific text.\n"
        f"11. Answer choices should contain only direct technical information without meta-references to content or sources.\n\n"
        f"Your response must follow this format precisely: \n"
        f"CONTEXT: <1-2 sentences establishing domain/topic context>\n"
        f"QUESTION: <the question>\n"
        f"1: <first answer choice>\n"
        f"2: <second answer choice>\n"
        f"...\n"
        f"CORRECT ANSWER: {target_correct_position}\n"
    )

    try:
        step2_start_time = time.time()
        response_2 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message_2},
                {"role": "user", "content": user_message_2},
            ],
            temperature=0.8,
        )
        step2_output = response_2['choices'][0]['message']['content'].strip()
        step2_time = time.time() - step2_start_time
    except Exception as e:
        log_message(f"Error generating question for chunk {chunk_id}: {e}", log_level="ERROR", error_type="question_gen")
        return {
            'chunk_id': chunk_id,
            'error': f"Error in step 2: {str(e)}",
            'status': 'error',
            'processing_time': time.time() - start_time
        }

    # --------------------------------------------------------------------
    # Step 3: Self-evaluate the generated question
    # --------------------------------------------------------------------
    system_message_3 = (
        "You are an expert teacher evaluating the quality of a multiple choice question. "
        "Your role is to ensure questions are clear, fair, and educationally valuable."
    )
    
    user_message_3 = (
        f"Evaluate the following multiple-choice question on a scale from 1-10, "
        f"where 10 is a perfect question.\n\n"
        f"CONTENT:\n{chunk_text}\n\n"
        f"QUESTION:\n{step2_output}\n\n"
        f"CONTENT RELEVANCE INFO:\n"
        f"- Relevance Score: {relevance_check['relevance_score']}/10\n"
        f"- Content Type: {relevance_check['content_type']}\n"
        f"- Relevance Reasoning: {relevance_check['reasoning']}\n\n"
        f"Rate the question based on these criteria:\n"
        f"- Clarity: Is the question clear and unambiguous?\n"
        f"- Accuracy: Is the content factually correct and aligned with the source material?\n"
        f"- Difficulty: Is the difficulty appropriate (challenging but fair)?\n"
        f"- Distractors: Are the incorrect options plausible but clearly wrong?\n"
        f"- Educational value: Does answering this question demonstrate understanding?\n"
        f"- Self-contained: CRITICAL - Does the question stand alone without ANY references to external materials?\n"
        f"- Content relevance: IMPORTANT - Questions based on low-relevance content (references, metadata, etc.) should receive lower scores\n\n"
        f"AUTOMATIC DISQUALIFIERS (score must be 1-3 if ANY are present):\n"
        f"- References to 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"- References to 'the author', 'according to', 'as mentioned', 'as described'\n"
        f"- References to 'Appendix', 'Figure', 'Table', 'Section', 'Chapter'\n"
        f"- References to 'above', 'below', 'previously mentioned', 'following'\n"
        f"- Any other references that assume the reader has access to external materials\n"
        f"- Content based primarily on references, copyright notices, or metadata (should score 1-4)\n\n"
        f"SCORING ADJUSTMENT FOR CONTENT RELEVANCE:\n"
        f"- If content relevance score is 1-4: Maximum question score should be 4\n"
        f"- If content relevance score is 5-7: Maximum question score should be 7\n"
        f"- If content relevance score is 8-10: Normal scoring applies\n\n"
        f"A truly self-contained question should read like a general knowledge question on the topic.\n\n"
        f"Provide your response in this format:\n"
        f"SCORE: <numeric score between 1-10>\n"
        f"CRITIQUE: <brief explanation of score>\n"
    )

    try:
        step3_start_time = time.time()
        response_3 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message_3},
                {"role": "user", "content": user_message_3},
            ],
            temperature=0.3,
        )
        step3_output = response_3['choices'][0]['message']['content'].strip()
        step3_time = time.time() - step3_start_time
        
        # Extract score from evaluation
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", step3_output)
        score = int(float(score_match.group(1))) if score_match else 0
        
        # Extract critique
        critique_match = re.search(r"CRITIQUE:(.*?)(?:\n\n|$)", step3_output, re.DOTALL)
        critique = critique_match.group(1).strip() if critique_match else "No critique provided"
        
    except Exception as e:
        log_message(f"Error evaluating question for chunk {chunk_id}: {e}", log_level="ERROR", error_type="question_eval")
        return {
            'chunk_id': chunk_id,
            'error': f"Error in step 3: {str(e)}",
            'status': 'error',
            'processing_time': time.time() - start_time
        }

    # Calculate total processing time
    processing_time = time.time() - start_time
    
    # If the score is below the threshold, don't return the question
    if score < min_score:
        log_message(f"Question for chunk {chunk_id} scored too low ({score}). Skipping.", log_level="WARNING", error_type="low_score")
        return {
            'chunk_id': chunk_id,
            'score': score,
            'critique': critique,
            'status': 'low_score',
            'processing_time': processing_time
        }
    
    # Extract context, question and answers from the model output
    context_match = re.search(r"CONTEXT:\s*(.*?)(?:\n|$)", step2_output, re.DOTALL)
    raw_context = context_match.group(1).strip() if context_match else ""
    context = clean_answer_content(raw_context)
    
    question_match = re.search(r"QUESTION:\s*(.*?)(?:\n|$)", step2_output)
    raw_question = question_match.group(1).strip() if question_match else "No question found"
    question = clean_answer_content(raw_question)
    
    # Extract all answer choices
    raw_answers = []
    for i in range(num_answers):
        answer_number = i + 1
        pattern = rf"{answer_number}:\s*(.*?)(?:\n|$)"
        answer_match = re.search(pattern, step2_output)
        if answer_match:
            raw_answers.append(answer_match.group(1).strip())
    
    # Clean the answer choices to remove any evaluation content
    answers = clean_answer_choices(raw_answers)
    
    # Use the target correct position we specified in the prompt
    # Verify that the AI followed our instruction by checking the response
    correct_answer_match = re.search(r"CORRECT ANSWER:\s*(\d+)", step2_output)
    ai_reported_answer = int(correct_answer_match.group(1)) if correct_answer_match else target_correct_position
    
    # Use our target position regardless of what AI reported (for uniform distribution)
    correct_answer_number = target_correct_position
    correct_index = correct_answer_number - 1  # Convert from 1-based to 0-based
    
    # Log if AI didn't follow our instruction (for debugging purposes)
    if ai_reported_answer != target_correct_position:
        log_message(f"Chunk {chunk_id}: AI reported answer {ai_reported_answer} but we specified {target_correct_position}", 
                   log_level="DEBUG")
    
    # Format the question with context and embedded choices like in NAT-MC.json
    formatted_question = ""
    if context:
        formatted_question = context + "\n\n" + question + "\n\n"
    else:
        formatted_question = question + "\n\n"
    for i, choice in enumerate(answers):
        choice_marker = "(*)" if i == correct_index else ""  # Mark correct answer with (*)
        formatted_question += f"{i+1}) {choice} {choice_marker}  \n"
    
    # Create clean answer explanation using numeric labels for consistency
    correct_number = correct_index + 1  # Convert 0-based index to 1-based number
    answer_explanation = f"The correct answer is {correct_number}) {answers[correct_index]}."
    
    # Prepare the result object with NAT-MC.json compatible format
    result = {
        'chunk_id': chunk_id,
        'question': formatted_question.strip(),
        'answer': answer_explanation,
        'text': chunk_text,
        'score': score,
        'critique': critique,
        'type': 'multiple-choice',  # hyphenated as in NAT-MC.json
        'status': 'completed',
        'processing_time': processing_time,
        'relevance_check': relevance_check,  # Store relevance information
        'step_times': {
            'summarize': step1_time,
            'generate': step2_time,
            'evaluate': step3_time
        }
    }
    
    return result


def generate_free_form_qa_pairs(chunk_id: str, chunk_text: str, model_name: str, 
                                min_score: int = 7) -> Dict:
    """
    Generate a free-form Q/A pair from a chunk.
    Returns a dictionary with question, answer, and other metadata.
    """
    # Start timing the processing
    start_time = time.time()
    
    # --------------------------------------------------------------------
    # Step 0: Check content relevance
    # --------------------------------------------------------------------
    relevance_check = check_content_relevance(chunk_text, model_name)
    
    # Skip non-relevant content
    if not relevance_check['is_relevant']:
        log_message(f"Chunk {chunk_id} skipped - not relevant to core content: {relevance_check['reasoning']}", 
                   log_level="INFO", error_type="content_filter")
        return {
            'chunk_id': chunk_id,
            'status': 'filtered_non_relevant',
            'processing_time': time.time() - start_time,
            'relevance_check': relevance_check,
            'message': f"Skipped non-relevant content: {relevance_check['content_type']}"
        }
    
    # --------------------------------------------------------------------
    # Step 1: Summarize & expand the chunk => augmented_chunk
    # --------------------------------------------------------------------
    system_message = (
        "You are a helpful assistant that summarizes text in bullet points "
        "and expands on them using your broader knowledge. "
        "Name this result 'augmented_chunk'."
    )
    user_message = (
        f"Given the following chunk of text, please:\n\n"
        f"1. Summarize the text in bullet points.\n"
        f"2. Expand on the summary using your parametric knowledge.\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        f"Return the result as plain text labeled 'augmented_chunk:' at the start."
    )

    try:
        step1_start_time = time.time()
        response_1 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
        )
        step1_output = response_1['choices'][0]['message']['content'].strip()
        step1_time = time.time() - step1_start_time
        
        # We'll assume the model starts with "augmented_chunk:"
        augmented_chunk = step1_output
        if "augmented_chunk:" in step1_output.lower():
            augmented_chunk = re.split(
                r'augmented_chunk\s*:\s*',
                step1_output,
                flags=re.IGNORECASE,
                maxsplit=1
            )[-1].strip()

    except Exception as e:
        log_message(f"Error summarizing chunk {chunk_id}: {e}", log_level="ERROR", error_type="summarizing")
        return {
            'chunk_id': chunk_id,
            'error': f"Error in step 1: {str(e)}",
            'status': 'error',
            'processing_time': time.time() - start_time
        }

    # --------------------------------------------------------------------
    # Step 2: Generate a question that can be answered by the augmented_chunk
    # --------------------------------------------------------------------
    system_message_2 = (
        "You are a helpful assistant that generates high-quality questions "
        "based on text provided by the user. Each question should test deep "
        "understanding and be specific enough to have a clear answer."
    )
    
    user_message_2 = (
        f"Generate ONE challenging question that would test a reader's deep understanding "
        f"of the content. The question should have a specific, clear answer.\n\n"
        f"Text:\n{augmented_chunk}\n\n"
        f"Requirements:\n"
        f"1. Begin with 1-2 sentences of contextual information that establishes the domain/topic without referencing source materials.\n"
        f"2. Create a question that probes deep understanding rather than simple recall.\n"
        f"3. Ensure the question has a clear, unambiguous answer.\n"
        f"4. The question should focus on important concepts or facts from the text.\n"
        f"5. Make it specific enough that the answer would demonstrate real comprehension.\n"
        f"6. The question should be concise and clearly worded.\n"
        f"7. CRITICAL: The context, question and answer must be completely self-contained. DO NOT reference any external materials including:\n"
        f"   - 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"   - 'the author states', 'according to the text', 'as mentioned', 'as described'\n"
        f"   - 'Appendix', 'Figure', 'Table', 'Section', 'Chapter', 'above', 'below'\n"
        f"   - Any other references to source materials or external content\n"
        f"8. The context and question should read as if testing general knowledge on the topic, not comprehension of a specific text.\n"
        f"9. The answer must contain only direct technical information and explanation without any meta-references to content, studies, or source materials.\n\n"
        f"Your response must follow this format precisely:\n"
        f"CONTEXT: <1-2 sentences establishing domain/topic context>\n"
        f"QUESTION: <the question>\n"
        f"ANSWER: <the complete, detailed answer with technical explanation only>\n"
    )

    try:
        step2_start_time = time.time()
        response_2 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message_2},
                {"role": "user", "content": user_message_2},
            ],
            temperature=0.8,
        )
        step2_output = response_2['choices'][0]['message']['content'].strip()
        step2_time = time.time() - step2_start_time
    except Exception as e:
        log_message(f"Error generating question for chunk {chunk_id}: {e}", log_level="ERROR", error_type="question_gen")
        return {
            'chunk_id': chunk_id,
            'error': f"Error in step 2: {str(e)}",
            'status': 'error',
            'processing_time': time.time() - start_time
        }

    # --------------------------------------------------------------------
    # Step 3: Self-evaluate the generated question
    # --------------------------------------------------------------------
    system_message_3 = (
        "You are an expert teacher evaluating the quality of a question-answer pair. "
        "Your role is to ensure questions are clear, fair, and educationally valuable."
    )
    
    user_message_3 = (
        f"Evaluate the following question and answer on a scale from 1-10, "
        f"where 10 is perfect.\n\n"
        f"CONTENT:\n{chunk_text}\n\n"
        f"QUESTION AND ANSWER:\n{step2_output}\n\n"
        f"CONTENT RELEVANCE INFO:\n"
        f"- Relevance Score: {relevance_check['relevance_score']}/10\n"
        f"- Content Type: {relevance_check['content_type']}\n"
        f"- Relevance Reasoning: {relevance_check['reasoning']}\n\n"
        f"Rate based on these criteria:\n"
        f"- Clarity: Is the question clear and unambiguous?\n"
        f"- Accuracy: Is the answer factually correct and aligned with the source material?\n"
        f"- Difficulty: Is the difficulty appropriate (challenging but fair)?\n"
        f"- Specificity: Does the question target specific understanding rather than general knowledge?\n"
        f"- Educational value: Does answering demonstrate meaningful understanding?\n"
        f"- Self-contained: CRITICAL - Does the question and answer stand alone without ANY references to external materials?\n"
        f"- Content relevance: IMPORTANT - Questions based on low-relevance content (references, metadata, etc.) should receive lower scores\n\n"
        f"AUTOMATIC DISQUALIFIERS (score must be 1-3 if ANY are present):\n"
        f"- References to 'the text', 'the passage', 'the document', 'the paper', 'the study'\n"
        f"- References to 'the author', 'according to', 'as mentioned', 'as described'\n"
        f"- References to 'Appendix', 'Figure', 'Table', 'Section', 'Chapter'\n"
        f"- References to 'above', 'below', 'previously mentioned', 'following'\n"
        f"- Any other references that assume the reader has access to external materials\n"
        f"- Content based primarily on references, copyright notices, or metadata (should score 1-4)\n\n"
        f"SCORING ADJUSTMENT FOR CONTENT RELEVANCE:\n"
        f"- If content relevance score is 1-4: Maximum question score should be 4\n"
        f"- If content relevance score is 5-7: Maximum question score should be 7\n"
        f"- If content relevance score is 8-10: Normal scoring applies\n\n"
        f"A truly self-contained question should read like a general knowledge question on the topic.\n\n"
        f"Provide your response in this format:\n"
        f"SCORE: <numeric score between 1-10>\n"
        f"CRITIQUE: <brief explanation of score>\n"
    )

    try:
        step3_start_time = time.time()
        response_3 = batched_openai_completion(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message_3},
                {"role": "user", "content": user_message_3},
            ],
            temperature=0.3,
        )
        step3_output = response_3['choices'][0]['message']['content'].strip()
        step3_time = time.time() - step3_start_time
        
        # Extract score from evaluation
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", step3_output)
        score = int(float(score_match.group(1))) if score_match else 0
        
        # Extract critique
        critique_match = re.search(r"CRITIQUE:(.*?)(?:\n\n|$)", step3_output, re.DOTALL)
        critique = critique_match.group(1).strip() if critique_match else "No critique provided"
        
    except Exception as e:
        log_message(f"Error evaluating question for chunk {chunk_id}: {e}", log_level="ERROR", error_type="question_eval")
        return {
            'chunk_id': chunk_id,
            'error': f"Error in step 3: {str(e)}",
            'status': 'error',
            'processing_time': time.time() - start_time
        }

    # Calculate total processing time
    processing_time = time.time() - start_time
    
    # If the score is below the threshold, don't return the question
    if score < min_score:
        log_message(f"Question for chunk {chunk_id} scored too low ({score}). Skipping.", log_level="WARNING", error_type="low_score")
        return {
            'chunk_id': chunk_id,
            'score': score,
            'critique': critique,
            'status': 'low_score',
            'processing_time': processing_time
        }
    
    # Extract context, question and answer from the model output
    context_match = re.search(r"CONTEXT:\s*(.*?)(?:\n|$)", step2_output, re.DOTALL)
    raw_context = context_match.group(1).strip() if context_match else ""
    context = clean_answer_content(raw_context)
    
    question_match = re.search(r"QUESTION:\s*(.*?)(?:\n|$)", step2_output, re.DOTALL)
    raw_question = question_match.group(1).strip() if question_match else "No question found"
    question = clean_answer_content(raw_question)
    
    answer_match = re.search(r"ANSWER:\s*(.*?)(?:\n\n|$)", step2_output, re.DOTALL)
    raw_answer = answer_match.group(1).strip() if answer_match else "No answer found"
    
    # Clean the answer to remove any evaluation commentary
    answer = clean_answer_content(raw_answer)
    
    # Format the question with context for consistency
    formatted_question = ""
    if context:
        formatted_question = context + "\n\n" + question
    else:
        formatted_question = question
    
    # Prepare the result object
    result = {
        'chunk_id': chunk_id,
        'question': formatted_question,
        'answer': answer,
        'text': chunk_text,
        'score': score,
        'critique': critique,
        'type': 'free_form',
        'status': 'completed',
        'processing_time': processing_time,
        'relevance_check': relevance_check,  # Store relevance information
        'step_times': {
            'summarize': step1_time,
            'generate': step2_time,
            'evaluate': step3_time
        }
    }
    
    return result

def generate_reasoning_trace_pairs(chunk_id: str, chunk_text: str, model_name: str, min_score: int = 7) -> Dict:
    """
    Generate a reasoning trace Q/A pair from a chunk using prompts adapted from SynthReasoner.
    Returns a dictionary with question, reasoning trace, answer, and other metadata.
    """
    global _exit_requested
    
    # Start timing the processing
    start_time = time.time()
    
    # Check for exit request
    if _exit_requested:
        return {
            'chunk_id': chunk_id,
            'status': 'cancelled',
            'processing_time': 0,
            'message': 'Cancelled due to shutdown request'
        }
    
    # --------------------------------------------------------------------
    # Step 0: Check content relevance
    # --------------------------------------------------------------------
    relevance_check = check_content_relevance(chunk_text, model_name)
    
    # Skip non-relevant content
    if not relevance_check['is_relevant']:
        log_message(f"Chunk {chunk_id} skipped - not relevant to core content: {relevance_check['reasoning']}", 
                   log_level="INFO", error_type="content_filter")
        return {
            'chunk_id': chunk_id,
            'status': 'filtered_non_relevant',
            'processing_time': time.time() - start_time,
            'relevance_check': relevance_check,
            'message': f"Skipped non-relevant content: {relevance_check['content_type']}"
        }
    
    try:
        # --------------------------------------------------------------------
        # Step 1: Generate challenging question
        # --------------------------------------------------------------------
        step1_start = time.time()
        
        question_prompt = f"""
You are an expert researcher specializing in creating challenging questions that test deep understanding and reasoning abilities. Your task is to generate ONE exceptional question about this content.

CONTENT:
{chunk_text}

TASK: Generate ONE challenging question that emphasizes logic, reasoning, and critical thinking.

QUESTION REQUIREMENTS:
1. DIFFICULTY: Should be challenging for intermediate to advanced researchers
2. LOGIC-ORIENTED: Requires systematic reasoning, not just recall
3. SELF-CONTAINED: Includes all necessary context within the question itself
4. REASONING-DEMANDING: Cannot be answered with simple factual lookup
5. INTELLECTUALLY RIGOROUS: Tests understanding of concepts, relationships, causality

PREFERRED QUESTION TYPES (choose the most challenging option):
- CRITICAL ANALYSIS: Challenge methodology, assumptions, or conclusions with specific technical concerns
- MECHANISTIC REASONING: How/why questions requiring step-by-step logical explanation
- LIMITATION ANALYSIS: Identify and explain constraints, boundaries, or failure modes
- COMPARATIVE REASONING: Contrast approaches, compare alternatives, or evaluate trade-offs
- APPLICATION CHALLENGES: Complex scenarios requiring multi-step reasoning to apply findings

QUESTION DESIGN PRINCIPLES:
- Start with sophisticated framing: "How would you reconcile...", "What explains the apparent contradiction between...", "Given the methodological constraints, how reliable is..."
- Require multi-step logical reasoning
- Force consideration of alternative explanations
- Demand evaluation of evidence quality
- Test understanding of causal relationships
- Challenge assumptions or oversimplifications
- Avoid references to the source text, e.g., "this paper", "the authors", "these results", "the thesis", "the model", etc.

FORMAT YOUR RESPONSE AS:
QUESTION_TYPE: [type from the preferred types above]
DIFFICULTY: advanced
QUESTION: [The challenging question text - should be substantial and sophisticated]

Focus on creating a question that would require an expert to engage in systematic, rigorous reasoning to answer properly.
"""
        
        messages = [{"role": "user", "content": question_prompt}]
        
        try:
            response = batched_openai_completion(model_name, messages)
            question_response = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            log_message(f"Error generating question for chunk {chunk_id}: {e}", log_level="ERROR", error_type="api")
            return {
                'chunk_id': chunk_id,
                'status': 'error_question_generation',
                'processing_time': time.time() - start_time,
                'message': f'Question generation error: {str(e)}'
            }
        
        step1_time = time.time() - step1_start
        
        # Parse question response
        question_data = {}
        lines = question_response.strip().split('\n')
        for line in lines:
            if line.startswith('QUESTION_TYPE:'):
                question_data['question_type'] = line.split(':', 1)[1].strip()
            elif line.startswith('DIFFICULTY:'):
                question_data['difficulty_level'] = line.split(':', 1)[1].strip()
            elif line.startswith('QUESTION:'):
                question_data['question'] = line.split(':', 1)[1].strip()
        
        if 'question' not in question_data:
            # Fallback: use the whole response as the question
            question_data['question'] = question_response.strip()
            question_data['question_type'] = 'critical_analysis'
            question_data['difficulty_level'] = 'advanced'
        
        # --------------------------------------------------------------------
        # Step 2: Generate reasoning trace and answer
        # --------------------------------------------------------------------
        step2_start = time.time()
        
        reasoning_prompt = f"""
You are an expert researcher providing a systematic analysis to answer a challenging question. Show rigorous, step-by-step reasoning that demonstrates scientific thinking at its best.

CONTENT:
{chunk_text}

QUESTION TO ANSWER:
Type: {question_data.get('question_type', 'critical_analysis')}
Difficulty: {question_data.get('difficulty_level', 'advanced')}
Question: {question_data['question']}

TASK: Provide a comprehensive reasoning trace followed by a clear final answer. In general this means:
1. fully decompose the question into parts
2. enumerate the parts to make sure you have understood the question in full
3. identify background information for every part of the question
4. double-check the background for accuracy, consistency, and relevance
5. in a logical chain, incorporate the background information in ways that lead to the final answer
6. double-check the final answer for accuracy, consistency, and relevance
7. if the final answer is not correct, revise the erroneous parts of the reasoning trace and rewrite the final answer

FORMAT YOUR RESPONSE AS:
REASONING:
<thought>
[Natural, flowing expert reasoning that systematically works through the question. Use conversational expert language as if you're thinking through a challenging problem out loud.

**CRITICAL STYLE REQUIREMENTS**:
- Write as a continuous stream of expert consciousness, NOT as structured analysis
- Use natural self-correction phrases woven into the flow: "Wait, let me reconsider...", "Actually, I need to correct this...", "On reflection, this assumption is flawed because..."
- NO structured headings, bullet points, or artificial phrases like "**Recognition of inconsistencies:**"
- Maintain logical progression while keeping it conversational and natural
- Show authentic expert thinking with genuine moments of doubt, correction, and insight]
</thought>

FINAL_ANSWER: [Clear, direct answer that addresses all aspects of the question while acknowledging appropriate uncertainties]

Emphasize natural expert thinking, logical rigor, and authentic self-correction within a conversational, flowing style.
"""
        
        messages = [{"role": "user", "content": reasoning_prompt}]
        
        try:
            response = batched_openai_completion(model_name, messages)
            reasoning_response = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            log_message(f"Error generating reasoning trace for chunk {chunk_id}: {e}", log_level="ERROR", error_type="api")
            return {
                'chunk_id': chunk_id,
                'status': 'error_reasoning_generation',
                'processing_time': time.time() - start_time,
                'message': f'Reasoning generation error: {str(e)}'
            }
        
        step2_time = time.time() - step2_start
        
        # Parse reasoning response
        reasoning_trace = ""
        final_answer = ""
        
        # Extract reasoning and final answer
        if "REASONING:" in reasoning_response and "FINAL_ANSWER:" in reasoning_response:
            parts = reasoning_response.split("REASONING:", 1)[1].split("FINAL_ANSWER:", 1)
            reasoning_trace = parts[0].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Fallback: use the whole response as reasoning trace
            reasoning_trace = reasoning_response.strip()
            final_answer = "Answer derived from the reasoning above."
        
        # Clean up reasoning trace (remove <thought> tags if present)
        if reasoning_trace.startswith("<thought>") and reasoning_trace.endswith("</thought>"):
            reasoning_trace = reasoning_trace[9:-10].strip()
        
        # --------------------------------------------------------------------
        # Step 3: Evaluate the reasoning trace quality
        # --------------------------------------------------------------------
        step3_start = time.time()
        
        evaluation_prompt = f"""
Rate the quality of this reasoning trace on a scale of 1-10, considering logical flow, depth of analysis, accuracy, and clarity.

QUESTION: {question_data['question']}
REASONING TRACE: {reasoning_trace}
FINAL ANSWER: {final_answer}

Provide a score (1-10) and brief critique explaining the strengths and weaknesses.

FORMAT:
SCORE: [number 1-10]
CRITIQUE: [brief explanation of the score]
"""
        
        messages = [{"role": "user", "content": evaluation_prompt}]
        
        try:
            response = batched_openai_completion(model_name, messages)
            evaluation_response = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            log_message(f"Error evaluating reasoning trace for chunk {chunk_id}: {e}", log_level="ERROR", error_type="api")
            # Use default values if evaluation fails
            score = 7
            critique = f"Evaluation failed: {str(e)}"
        else:
            # Parse evaluation response
            score = 7  # default
            critique = evaluation_response.strip()
            
            for line in evaluation_response.split('\n'):
                if line.startswith('SCORE:'):
                    try:
                        score = int(line.split(':', 1)[1].strip())
                    except ValueError:
                        score = 7
                elif line.startswith('CRITIQUE:'):
                    critique = line.split(':', 1)[1].strip()
        
        step3_time = time.time() - step3_start
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Log successful generation
        log_message(f"Generated reasoning trace for chunk {chunk_id} (score: {score})")
        
        # Check if score meets minimum threshold
        if score < min_score:
            log_message(f"Reasoning trace for chunk {chunk_id} scored {score} (below minimum {min_score})", 
                       log_level="INFO", error_type="low_score")
            return {
                'chunk_id': chunk_id,
                'status': 'low_score',
                'processing_time': processing_time,
                'score': score,
                'message': f'Score {score} below minimum {min_score}'
            }
        
        # Prepare the result object
        result = {
            'chunk_id': chunk_id,
            'question': question_data['question'],
            'question_type': question_data.get('question_type', 'critical_analysis'),
            'difficulty_level': question_data.get('difficulty_level', 'advanced'),
            'reasoning_trace': reasoning_trace,
            'answer': final_answer,
            'text': chunk_text,
            'score': score,
            'critique': critique,
            'type': 'reasoning_trace',
            'status': 'completed',
            'processing_time': processing_time,
            'relevance_check': relevance_check,
            'step_times': {
                'question_gen': step1_time,
                'reasoning_gen': step2_time,
                'evaluate': step3_time
            }
        }
        
        return result
        
    except Exception as e:
        # Handle any unexpected errors
        log_message(f"Unexpected error generating reasoning trace for chunk {chunk_id}: {e}", 
                   log_level="ERROR", error_type="other")
        return {
            'chunk_id': chunk_id,
            'status': 'error_unexpected',
            'processing_time': time.time() - start_time,
            'message': f'Unexpected error: {str(e)}'
        }

def process_chunks_with_parallel_workers(chunk_ids: List[str], chunks_dir: str, model_name: str,
                            question_type: QuestionType, num_answers: int, min_score: int,
                            checkpoint_manager, output_file: str):
    """
    Process chunks in parallel using a worker pool.
    Master process handles all I/O operations.
    Workers only perform question generation.
    
    Args:
        chunk_ids: List of chunk IDs to process
        chunks_dir: Directory containing chunk files
        model_name: Name of the model to use
        question_type: Type of questions to generate
        num_answers: Number of answers for multiple-choice questions
        min_score: Minimum score for keeping a question
        checkpoint_manager: Checkpoint manager instance
        output_file: Path to output file for saving results
        
    Returns:
        List of successful question-answer pairs
    """
    global _exit_requested, _processed_chunks, terminal_ui, _max_workers
    
    # Check if any chunks need processing
    if not chunk_ids:
        log_message("No chunks to process.")
        return []
    
    # Filter out already processed chunks
    if checkpoint_manager:
        chunk_ids = [c_id for c_id in chunk_ids if not checkpoint_manager.is_chunk_processed(c_id)]
    
    if not chunk_ids:
        log_message("All chunks have already been processed.")
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(status_message="All chunks already processed")
        return []
    
    # Shuffle chunks for more diverse processing
    random.shuffle(chunk_ids)
    total_chunks = len(chunk_ids)
    
    # Initialize worker pool if not already done
    if not _worker_pool:
        init_worker_pool(_max_workers)
    
    log_message(f"Processing {total_chunks} chunks in parallel with {_max_workers} workers")
    
    # Results list for completed questions
    results = []
    successful_questions = 0
    
    # Start timing
    start_time = time.time()
    last_estimate_time = start_time
    estimate_interval = 10  # Update estimates more frequently with the UI
    
    # Setup progress bar for non-UI mode only
    if not _use_split_screen or not terminal_ui:
        progress_bar = tqdm(
            total=total_chunks,
            desc="Processing chunks",
            unit="chunk",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    # Dictionary to track futures and their corresponding chunk_ids
    futures = {}
    
    # Maximum number of pending futures to avoid overwhelming memory
    # This effectively creates a sliding window of active jobs
    max_pending_futures = _max_workers * 2  # Allow for some extra pending tasks
    
    # Track current chunk index for progress reporting
    current_chunk_idx = 0
    
    # Process chunks in batches
    while current_chunk_idx < len(chunk_ids) and not _exit_requested:
        # Submit new chunks to the worker pool up to max_pending_futures
        while len(futures) < max_pending_futures and current_chunk_idx < len(chunk_ids):
            chunk_id = chunk_ids[current_chunk_idx]
            current_chunk_idx += 1
            
            # Check for exit request
            if _exit_requested:
                break
                
            # Read the chunk file - this is I/O handled by the master process
            chunk_subdir = chunk_id[:2]
            chunk_file_path = os.path.join(chunks_dir, chunk_subdir, f"{chunk_id}.txt")
            
            try:
                with open(chunk_file_path, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
            except Exception as e:
                log_message(f"Error reading chunk {chunk_id}: {e}", log_level="ERROR", error_type="chunk_reading")
                
                # Update chunk status on error
                if checkpoint_manager:
                    checkpoint_manager.update_processed_chunk(chunk_id, {
                        'status': 'error',
                        'error': f"Error reading chunk: {str(e)}",
                        'error_time': time.time()
                    })
                continue
            
            # Submit chunk to worker pool
            log_message(f"Submitting chunk {chunk_id} to worker pool")
            future = submit_chunk_to_worker_pool(
                chunk_id, chunk_text, model_name, question_type, num_answers, min_score
            )
            futures[future] = chunk_id
            
            # Update UI
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(
                    current_chunk=chunk_id,
                    status_message=f"Submitted {current_chunk_idx}/{total_chunks} chunks, {len(futures)} in progress"
                )
                update_global_stats()
        
        # Process completed futures
        completed_futures = []
        for future in list(futures.keys()):
            if future.done():
                chunk_id = futures[future]
                completed_futures.append((future, chunk_id))
        
        # Process completed chunks
        for future, chunk_id in completed_futures:
            try:
                # Get result from future
                result = future.result()
                
                # Remove from tracking
                del futures[future]
                
                # Update counters
                with _counter_lock:
                    _processed_chunks += 1
                
                # Handle the result - this is I/O handled by the master process
                if result:
                    # Update progress bar in non-UI mode
                    if not _use_split_screen or not terminal_ui:
                        progress_bar.update(1)
                    
                    # Save to checkpoint
                    if checkpoint_manager:
                        checkpoint_manager.update_processed_chunk(chunk_id, result)
                    
                    # Handle successful results
                    if result.get('status') == 'completed':
                        successful_questions += 1
                        results.append(result)
                        
                        # Update progress/success rate
                        success_rate = (successful_questions / _processed_chunks) * 100
                        
                        # Update progress indicator
                        if not _use_split_screen or not terminal_ui:
                            progress_bar.set_description(
                                f"Processing chunks - {successful_questions} Q's generated ({success_rate:.1f}% success rate)"
                            )
                        else:
                            # Update UI with the new statistics
                            terminal_ui.update_stats(
                                success_rate=success_rate,
                                questions_generated=successful_questions,
                                chunks_processed=_processed_chunks
                            )
                        
                        # Log successful generation
                        if question_type == QuestionType.REASONING_TRACE:
                            log_message(f"Generated trace from chunk {chunk_id} (Trace #{successful_questions})")
                        else:
                            log_message(f"Generated question from chunk {chunk_id} (Question #{successful_questions})")
                        
                        # Create question data for output file - using NAT-MC.json format
                        question_data = {
                            'question': result.get('question', ''),
                            'answer': result.get('answer', ''),
                            'text': result.get('text', ''),
                            'type': result.get('type', ''),
                        }
                        
                        # Add reasoning trace for reasoning trace mode
                        if result.get('type') == 'reasoning_trace':
                            question_data['reasoning_trace'] = result.get('reasoning_trace', '')
                        
                        # Append to output file immediately
                        try:
                            # Load existing questions if the file exists
                            existing_questions = []
                            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    try:
                                        loaded_data = json.load(f)
                                        # Ensure loaded data is a list
                                        if isinstance(loaded_data, list):
                                            existing_questions = loaded_data
                                        else:
                                            log_message(f"Warning: Output file {output_file} contains unexpected format, creating new file", 
                                                      log_level="WARNING")
                                            existing_questions = []
                                    except json.JSONDecodeError:
                                        log_message(f"Warning: Could not parse existing output file {output_file}, creating new file", 
                                                  log_level="WARNING")
                            
                            # Combine existing questions with the new one
                            all_questions = existing_questions + [question_data]
                            
                            # Write to the output file
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(all_questions, f, ensure_ascii=False, indent=2)
                            
                        except Exception as e:
                            log_message(f"Error writing to output file: {e}", log_level="ERROR")
                    else:
                        # Log unsuccessful attempt with reason
                        status = result.get('status', 'unknown')
                        log_message(f"Chunk {chunk_id} processed but no question generated (status: {status})")
            
            except Exception as e:
                # Error handling for future.result()
                log_message(f"Error processing result for chunk {chunk_id}: {e}", log_level="ERROR")
                
                # Update chunk status on error
                if checkpoint_manager:
                    checkpoint_manager.update_processed_chunk(chunk_id, {
                        'status': 'error',
                        'error': f"Error processing result: {str(e)}",
                        'error_time': time.time()
                    })
        
        # Update time estimates periodically
        current_time = time.time()
        if current_time - last_estimate_time >= estimate_interval:
            last_estimate_time = current_time
            
            # Calculate completion percentage
            completion_percent = (_processed_chunks / total_chunks * 100)
            
            # Update UI
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(
                    completion_percentage=completion_percent,
                    status_message=f"Processed {_processed_chunks}/{total_chunks} chunks, {len(futures)} in progress"
                )
            
            # Always make sure to update global stats
            update_global_stats()
        
        # Short sleep to prevent CPU hogging if no futures are ready
        if not completed_futures:
            time.sleep(0.1)
    
    # Wait for remaining futures to complete
    log_message(f"Waiting for {len(futures)} remaining chunks to complete...")
    
    # Continue processing remaining futures
    while futures and not _exit_requested:
        # Process completed futures
        completed_futures = []
        for future in list(futures.keys()):
            if future.done():
                chunk_id = futures[future]
                completed_futures.append((future, chunk_id))
        
        # Process completed chunks (same code as above)
        for future, chunk_id in completed_futures:
            try:
                # Get result from future
                result = future.result()
                
                # Remove from tracking
                del futures[future]
                
                # Update counters
                with _counter_lock:
                    _processed_chunks += 1
                
                # Handle the result
                if result:
                    # Update progress bar in non-UI mode
                    if not _use_split_screen or not terminal_ui:
                        progress_bar.update(1)
                    
                    # Save to checkpoint
                    if checkpoint_manager:
                        checkpoint_manager.update_processed_chunk(chunk_id, result)
                    
                    # Handle successful results
                    if result.get('status') == 'completed':
                        successful_questions += 1
                        results.append(result)
                        
                        # Update progress/success rate
                        success_rate = (successful_questions / _processed_chunks) * 100
                        
                        # Update progress indicator
                        if not _use_split_screen or not terminal_ui:
                            progress_bar.set_description(
                                f"Processing chunks - {successful_questions} Q's generated ({success_rate:.1f}% success rate)"
                            )
                        else:
                            # Update UI with the new statistics
                            terminal_ui.update_stats(
                                success_rate=success_rate,
                                questions_generated=successful_questions,
                                chunks_processed=_processed_chunks
                            )
                        
                        # Log successful generation
                        if question_type == QuestionType.REASONING_TRACE:
                            log_message(f"Generated trace from chunk {chunk_id} (Trace #{successful_questions})")
                        else:
                            log_message(f"Generated question from chunk {chunk_id} (Question #{successful_questions})")
                        
                        # Create question data for output file - using NAT-MC.json format
                        question_data = {
                            'question': result.get('question', ''),
                            'answer': result.get('answer', ''),
                            'text': result.get('text', ''),
                            'type': result.get('type', ''),
                        }
                        
                        # Add reasoning trace for reasoning trace mode
                        if result.get('type') == 'reasoning_trace':
                            question_data['reasoning_trace'] = result.get('reasoning_trace', '')
                        
                        # Append to output file immediately
                        try:
                            # Load existing questions if the file exists
                            existing_questions = []
                            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                                with open(output_file, 'r', encoding='utf-8') as f:
                                    try:
                                        loaded_data = json.load(f)
                                        # Ensure loaded data is a list
                                        if isinstance(loaded_data, list):
                                            existing_questions = loaded_data
                                        else:
                                            log_message(f"Warning: Output file {output_file} contains unexpected format, creating new file", 
                                                      log_level="WARNING")
                                            existing_questions = []
                                    except json.JSONDecodeError:
                                        log_message(f"Warning: Could not parse existing output file {output_file}, creating new file", 
                                                  log_level="WARNING")
                            
                            # Combine existing questions with the new one
                            all_questions = existing_questions + [question_data]
                            
                            # Write to the output file
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(all_questions, f, ensure_ascii=False, indent=2)
                            
                        except Exception as e:
                            log_message(f"Error writing to output file: {e}", log_level="ERROR")
                    else:
                        # Log unsuccessful attempt with reason
                        status = result.get('status', 'unknown')
                        log_message(f"Chunk {chunk_id} processed but no question generated (status: {status})")
            
            except Exception as e:
                # Error handling for future.result()
                log_message(f"Error processing result for chunk {chunk_id}: {e}", log_level="ERROR")
                
                # Update chunk status on error
                if checkpoint_manager:
                    checkpoint_manager.update_processed_chunk(chunk_id, {
                        'status': 'error',
                        'error': f"Error processing result: {str(e)}",
                        'error_time': time.time()
                    })
        
        # Update time estimates
        update_global_stats()
        
        # Short sleep to prevent CPU hogging if no futures are ready
        if not completed_futures:
            time.sleep(0.1)
    
    # Clean up progress bar in non-UI mode
    if not _use_split_screen or not terminal_ui:
        progress_bar.close()
    
    # Final progress report
    elapsed_time = time.time() - start_time
    
    completion_message = (
        f"\nCompleted processing {_processed_chunks}/{total_chunks} chunks in {human_readable_time(elapsed_time)}, "
        f"generated {successful_questions} questions"
    )
    log_message(completion_message)
    
    if terminal_ui and _use_split_screen:
        terminal_ui.update_stats(
            status_message=f"Completed! Generated {successful_questions} questions"
        )
        update_global_stats()
    
    return results


##############################################################################
# Command-line argument parsing
##############################################################################
##############################################################################
# Checkpoint Manager
##############################################################################

class CheckpointManager:
    """
    Checkpoint manager for tracking processed files and questions.
    """
    _instance = None  # Class variable to store singleton instance

    @classmethod
    def get_instance(cls) -> Optional['CheckpointManager']:
        """
        Get the current singleton instance of the CheckpointManager.
        Returns None if no instance has been created yet.
        """
        return cls._instance

    def __init__(self, checkpoint_file: str, force_restart: bool = False):
        self.checkpoint_file = checkpoint_file
        self.last_save_time = 0
        self.save_interval = 10  # Save at least every 10 seconds for better progress tracking
        self.question_type = None  # Will be set later via set_question_type()

        # Store this instance as the singleton
        CheckpointManager._instance = self

        # Initialize with empty data or load existing checkpoint
        if force_restart:
            self.checkpoint_data = {
                'processed_files': {},
                'processed_chunks': {},
                'questions': [],
                'error_stats': {
                    'error_file_processing': 0,
                    'error_chunk_extraction': 0,
                    'error_chunk_reading': 0,
                    'error_summarizing': 0,
                    'error_question_gen': 0,
                    'error_question_eval': 0,
                    'error_api': 0,
                    'error_other': 0,
                    'low_score_questions': 0,
                    'total_errors': 0
                },
                'counters': {
                    'total_files': 0,
                    'processed_files': 0,
                    'total_chunks': 0,
                    'processed_chunks': 0,
                    'extracted_chunks': 0
                }
            }
            log_message("Force restart: Starting with empty checkpoint")
        else:
            self.checkpoint_data = self._load_checkpoint()

    def _ensure_counters_exist(self):
        """
        Ensure that the counters dictionary exists in the checkpoint data.
        This is called before any operation that accesses the counters.
        """
        if 'counters' not in self.checkpoint_data:
            # Initialize counters with default values
            self.checkpoint_data['counters'] = {
                'total_files': 0,
                'processed_files': 0,
                'total_chunks': 0,
                'processed_chunks': 0,
                'extracted_chunks': 0
            }
            log_message("Initialized missing counters in checkpoint data", log_level="WARNING")

    def _load_checkpoint(self):
        """
        Load checkpoint data from JSON file if it exists.
        Returns a dictionary with processed files, processed chunks, and questions generated.
        """
        checkpoint_path = pathlib.Path(self.checkpoint_file)
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    
                    # Ensure the checkpoint has the required structure
                    if 'processed_chunks' not in checkpoint_data:
                        checkpoint_data['processed_chunks'] = {}
                        
                    # Add error_stats section if missing (for backward compatibility)
                    if 'error_stats' not in checkpoint_data:
                        checkpoint_data['error_stats'] = {
                            'error_file_processing': 0,
                            'error_chunk_extraction': 0,
                            'error_chunk_reading': 0,
                            'error_summarizing': 0,
                            'error_question_gen': 0,
                            'error_question_eval': 0,
                            'error_api': 0,
                            'error_other': 0,
                            'low_score_questions': 0,
                            'total_errors': 0
                        }
                    
                    # Add counters section if missing (for backward compatibility)
                    if 'counters' not in checkpoint_data:
                        # Calculate counters from the existing data
                        processed_files_count = sum(1 for f in checkpoint_data['processed_files'].values() 
                                                if f.get('status') == 'chunked')
                        total_chunks_count = len(checkpoint_data['processed_chunks'])
                        processed_chunks_count = sum(1 for c in checkpoint_data['processed_chunks'].values()
                                                if c.get('status') in ['completed', 'low_score', 'error'])
                        extracted_chunks_count = sum(1 for c in checkpoint_data['processed_chunks'].values()
                                                if c.get('status') == 'extracted')
                        
                        checkpoint_data['counters'] = {
                            'total_files': len(checkpoint_data['processed_files']),
                            'processed_files': processed_files_count,
                            'total_chunks': total_chunks_count,
                            'processed_chunks': processed_chunks_count,
                            'extracted_chunks': extracted_chunks_count
                        }
                        
                    # Count fully processed files (only those marked as 'chunked')
                    processed_count = sum(1 for f in checkpoint_data['processed_files'].values() 
                                     if f.get('status') == 'chunked')
                                     
                    # Calculate total errors from checkpoint for reporting
                    total_errors = checkpoint_data['error_stats'].get('total_errors', 0)
                    
                    # Get counter stats (or calculate them if not present)
                    if 'counters' in checkpoint_data:
                        counter_stats = checkpoint_data['counters']
                    else:
                        # Calculate from the data
                        counter_stats = {
                            'total_files': len(checkpoint_data['processed_files']),
                            'processed_files': processed_count,
                            'total_chunks': len(checkpoint_data['processed_chunks']),
                            'processed_chunks': sum(1 for c in checkpoint_data['processed_chunks'].values()
                                               if c.get('status') in ['completed', 'low_score', 'error']),
                            'extracted_chunks': sum(1 for c in checkpoint_data['processed_chunks'].values()
                                               if c.get('status') == 'extracted')
                        }
                                     
                    # Use log_message to ensure output appears in the correct pane
                    log_message(f"Loaded checkpoint: {processed_count} files already processed out of {counter_stats['total_files']} total files")
                    log_message(f"Checkpoint contains {len(checkpoint_data['processed_chunks'])} chunks: {counter_stats['extracted_chunks']} extracted, {counter_stats['processed_chunks']} processed")
                    log_message(f"Checkpoint contains {len(checkpoint_data['questions'])} previously generated questions")
                    log_message(f"Checkpoint contains {total_errors} tracked errors")
                    return checkpoint_data
            except Exception as e:
                # Use log_message for error output
                log_message(f"Error loading checkpoint file: {e}", log_level="ERROR")
                return {
                    'processed_files': {}, 
                    'processed_chunks': {}, 
                    'questions': [],
                    'error_stats': {
                        'error_file_processing': 0,
                        'error_chunk_extraction': 0,
                        'error_chunk_reading': 0,
                        'error_summarizing': 0,
                        'error_question_gen': 0,
                        'error_question_eval': 0,
                        'error_api': 0,
                        'error_other': 0,
                        'low_score_questions': 0,
                        'total_errors': 0
                    },
                    'counters': {
                        'total_files': 0,
                        'processed_files': 0,
                        'total_chunks': 0,
                        'processed_chunks': 0,
                        'extracted_chunks': 0
                    }
                }
        else:
            log_message("No checkpoint found, starting from beginning")
            return {
                'processed_files': {}, 
                'processed_chunks': {}, 
                'questions': [],
                'error_stats': {
                    'error_file_processing': 0,
                    'error_chunk_extraction': 0,
                    'error_chunk_reading': 0,
                    'error_summarizing': 0,
                    'error_question_gen': 0,
                    'error_question_eval': 0,
                    'error_api': 0,
                    'error_other': 0,
                    'low_score_questions': 0,
                    'total_errors': 0
                },
                'counters': {
                    'total_files': 0,
                    'processed_files': 0,
                    'total_chunks': 0,
                    'processed_chunks': 0,
                    'extracted_chunks': 0
                }
            }

    def force_save(self):
        """
        Force an immediate save of the checkpoint data to disk, regardless of timing.
        """
        try:
            # Create a temporary file first, then rename to avoid corruption
            temp_file = f"{self.checkpoint_file}.temp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, ensure_ascii=False, indent=2)

            # Rename the temp file to the actual checkpoint file
            # This is more atomic and helps prevent corrupted files
            import os
            if os.path.exists(self.checkpoint_file):
                os.replace(temp_file, self.checkpoint_file)
            else:
                os.rename(temp_file, self.checkpoint_file)

            # Count processed files (only those marked as 'chunked')
            processed_count = sum(1 for f in self.checkpoint_data['processed_files'].values() 
                              if f.get('status') == 'chunked')
            
            chunks_count = len(self.checkpoint_data['processed_chunks'])
            questions_count = len(self.checkpoint_data['questions'])
            
            # Use log_message instead of print to ensure output goes to the correct pane
            if self.question_type and hasattr(self.question_type, 'value') and self.question_type.value == "rt":
                label = "traces"
            else:
                label = "questions"
            log_message(f"Checkpoint saved: {processed_count} files, {chunks_count} chunks, {questions_count} {label}")

            # Update last save time
            self.last_save_time = time.time()
            return True
        except Exception as e:
            # Use log_message to ensure error appears in the correct pane
            log_message(f"Error saving checkpoint: {e}", log_level="ERROR")
            return False

    def save(self):
        """
        Save checkpoint data to JSON file if enough time has passed.
        """
        current_time = time.time()
        # Only save if we haven't saved recently
        if current_time - self.last_save_time >= self.save_interval:
            return self.force_save()
        return True

    def update_processed_chunk(self, chunk_id: str, chunk_data: dict):
        """
        Update a processed chunk in the checkpoint.
        """
        # Check if this is a status change
        status_changed = False
        old_status = "none"
        if chunk_id in self.checkpoint_data['processed_chunks']:
            old_status = self.checkpoint_data['processed_chunks'][chunk_id].get('status', 'none')
            new_status = chunk_data.get('status', 'none')
            status_changed = old_status != new_status
        
        # Update the chunk data
        self.checkpoint_data['processed_chunks'][chunk_id] = chunk_data
        
        # Ensure counters exist before updating them
        self._ensure_counters_exist()
        
        # Update counters based on status changes
        new_status = chunk_data.get('status', 'none')
        if status_changed:
            # If changing from 'extracted' to a processed status, increment processed_chunks
            if old_status == 'extracted' and new_status in ['completed', 'low_score', 'error']:
                self.checkpoint_data['counters']['processed_chunks'] += 1
                # Also decrement extracted_chunks since it's no longer just extracted
                self.checkpoint_data['counters']['extracted_chunks'] = max(0, self.checkpoint_data['counters']['extracted_chunks'] - 1)
            # If this is a new chunk being marked as extracted
            elif old_status == 'none' and new_status == 'extracted':
                self.checkpoint_data['counters']['extracted_chunks'] += 1

        # If this resulted in a completed question, add it to questions
        if chunk_data.get('status') == 'completed':
            # Create a question entry from the chunk data
            question_data = {
                'question': chunk_data.get('question', ''),
                'answer': chunk_data.get('answer', ''),
                'text': chunk_data.get('text', ''),
                'type': chunk_data.get('type', ''),
                'score': chunk_data.get('score', 0),
                'chunk_id': chunk_id,
                'processing_time': chunk_data.get('processing_time', 0)
            }

            # Add to questions list
            self.checkpoint_data['questions'].append(question_data)
            
        # Update total chunks count
        self.checkpoint_data['counters']['total_chunks'] = len(self.checkpoint_data['processed_chunks'])

        # Force save after every update to ensure no chunks are lost
        self.force_save()

    def add_processed_file(self, file_id: str, file_data: dict, chunk_ids: list):
        """
        Add a processed file to the checkpoint.
        """
        # Update processed files
        self.checkpoint_data['processed_files'][file_id] = file_data
        
        # Ensure counters exist before updating them
        self._ensure_counters_exist()
        
        # Initialize chunk entries if they don't exist
        for chunk_id in chunk_ids:
            if chunk_id not in self.checkpoint_data['processed_chunks']:
                self.checkpoint_data['processed_chunks'][chunk_id] = {
                    'file_id': file_id,
                    'status': 'extracted',
                    'extraction_time': time.time()
                }
                
                # Update extracted chunks counter
                self.checkpoint_data['counters']['extracted_chunks'] += 1
        
        # Update file counters
        if file_data.get('status') == 'chunked':
            # This is a fully processed file
            self.checkpoint_data['counters']['processed_files'] += 1
            
        # Ensure total_files is at least the number of files in processed_files
        self.checkpoint_data['counters']['total_files'] = max(
            self.checkpoint_data['counters']['total_files'],
            len(self.checkpoint_data['processed_files'])
        )
            
        # Update total chunks counter
        self.checkpoint_data['counters']['total_chunks'] = len(self.checkpoint_data['processed_chunks'])
        
        # Save checkpoint after each update
        self.save()

    def is_chunk_processed(self, chunk_id: str) -> bool:
        """
        Check if a chunk has already been processed.
        """
        return (chunk_id in self.checkpoint_data['processed_chunks'] and 
               self.checkpoint_data['processed_chunks'][chunk_id].get('status') in ['completed', 'low_score'])

    def get_unprocessed_chunks(self) -> List[str]:
        """
        Get a list of chunk IDs that have been extracted but not yet processed.
        """
        return [chunk_id for chunk_id, chunk_data in self.checkpoint_data['processed_chunks'].items()
               if chunk_data.get('status') == 'extracted']

    def get_processed_files(self) -> dict:
        """
        Get a copy of the processed files dictionary.
        """
        return dict(self.checkpoint_data['processed_files'])

    def get_processed_chunks(self) -> dict:
        """
        Get a copy of the processed chunks dictionary.
        """
        return dict(self.checkpoint_data['processed_chunks'])

    def get_questions(self) -> list:
        """
        Get a copy of the questions list.
        """
        return list(self.checkpoint_data['questions'])

    def get_questions_count(self) -> int:
        """
        Get the current number of questions.
        """
        return len(self.checkpoint_data['questions'])
        
    def set_question_type(self, question_type):
        """
        Set the question type for display purposes.
        """
        self.question_type = question_type
        
    def get_error_stats(self) -> dict:
        """
        Get the error statistics from the checkpoint.
        """
        return dict(self.checkpoint_data.get('error_stats', {
            'error_file_processing': 0,
            'error_chunk_extraction': 0,
            'error_chunk_reading': 0,
            'error_summarizing': 0,
            'error_question_gen': 0,
            'error_question_eval': 0,
            'error_api': 0,
            'error_other': 0,
            'low_score_questions': 0,
            'total_errors': 0
        }))
        
    def get_counter_stats(self) -> dict:
        """
        Get the counter statistics from the checkpoint.
        """
        return dict(self.checkpoint_data.get('counters', {
            'total_files': 0,
            'processed_files': 0,
            'total_chunks': 0,
            'processed_chunks': 0,
            'extracted_chunks': 0
        }))

    def get_completion_stats(self) -> dict:
        """
        Get statistics about completion progress.
        """
        total_chunks = len(self.checkpoint_data['processed_chunks'])
        completed_chunks = sum(1 for c in self.checkpoint_data['processed_chunks'].values()
                           if c.get('status') in ['completed', 'low_score'])
        error_chunks = sum(1 for c in self.checkpoint_data['processed_chunks'].values()
                       if c.get('status') == 'error')
        
        return {
            'total_chunks': total_chunks,
            'completed_chunks': completed_chunks,
            'error_chunks': error_chunks,
            'remaining_chunks': total_chunks - completed_chunks - error_chunks,
            'completion_percentage': min(100.0, (completed_chunks / total_chunks * 100)) if total_chunks > 0 else 0
        }


def display_exit_summary(stats, error_stats, completion_percentage, chunk_percentage, checkpoint_manager):
    """
    Display a comprehensive exit summary with all processing statistics.
    """
    total_processing_time = time.time() - _start_time
    
    # Determine completion reason
    completion_reason = "Normal completion"
    if _exit_requested:
        completion_reason = "Process interrupted (user request or error threshold exceeded)"
    
    # Determine completion status
    all_files_processed = (_processed_files >= _total_files)
    all_chunks_processed = (stats['completed_chunks'] >= stats['total_chunks'])
    has_errors = (error_stats['total_errors'] > 0)
    
    completion_status = "COMPLETE"
    if not all_files_processed or not all_chunks_processed:
        completion_status = "PARTIAL"
    if has_errors:
        completion_status += " (with errors)"
    
    # Display comprehensive summary
    log_message("\n" + "=" * 80, log_level="WARNING")
    log_message("FINAL PROCESSING SUMMARY", log_level="WARNING")
    log_message("=" * 80, log_level="WARNING")
    
    # Overall status
    log_message(f"\nSTATUS: {completion_status}", log_level="WARNING")
    log_message(f"COMPLETION REASON: {completion_reason}", log_level="WARNING")
    log_message(f"TOTAL PROCESSING TIME: {human_readable_time(total_processing_time)}", log_level="WARNING")
    
    # Processing statistics
    log_message("\nPROCESSING STATISTICS:", log_level="WARNING")
    log_message(f"  Papers/Files Processed: {_processed_files:,} / {_total_files:,} ({completion_percentage:.1f}%)", log_level="WARNING")
    log_message(f"  Chunks Extracted: {_extracted_chunks:,}", log_level="WARNING")
    log_message(f"  Chunks Processed: {stats['completed_chunks']:,} / {stats['total_chunks']:,} ({chunk_percentage:.1f}%)", log_level="WARNING")
    if question_type == QuestionType.REASONING_TRACE:
        log_message(f"  Traces Generated: {checkpoint_manager.get_questions_count():,}", log_level="WARNING")
    else:
        log_message(f"  Questions Generated: {checkpoint_manager.get_questions_count():,}", log_level="WARNING")
    
    # Error summary
    log_message("\nERROR SUMMARY:", log_level="WARNING")
    log_message(f"  Total Errors: {error_stats['total_errors']:,}", log_level="WARNING")
    if error_stats['total_errors'] > 0:
        log_message(f"    File Processing Errors: {error_stats['error_file_processing']:,}", log_level="WARNING")
        log_message(f"    Chunk Extraction Errors: {error_stats['error_chunk_extraction']:,}", log_level="WARNING")
        log_message(f"    Chunk Reading Errors: {error_stats['error_chunk_reading']:,}", log_level="WARNING")
        log_message(f"    Summarizing Errors: {error_stats['error_summarizing']:,}", log_level="WARNING")
        log_message(f"    Question Generation Errors: {error_stats['error_question_gen']:,}", log_level="WARNING")
        log_message(f"    Question Evaluation Errors: {error_stats['error_question_eval']:,}", log_level="WARNING")
        log_message(f"    API Errors: {error_stats['error_api']:,}", log_level="WARNING")
        log_message(f"    Low Score Questions: {error_stats['low_score_questions']:,}", log_level="WARNING")
        log_message(f"    Other Errors: {error_stats['error_other']:,}", log_level="WARNING")
    else:
        log_message(f"  No errors encountered! \u2713", log_level="WARNING")
    
    # Data completeness assessment
    log_message("\nDATA COMPLETENESS ASSESSMENT:", log_level="WARNING")
    if all_files_processed and all_chunks_processed and not has_errors:
        log_message("  \u2713 ALL DATA SUCCESSFULLY PROCESSED", log_level="WARNING")
    else:
        if not all_files_processed:
            remaining_files = _total_files - _processed_files
            log_message(f"  \u26a0️ {remaining_files:,} files remain unprocessed", log_level="WARNING")
        if not all_chunks_processed:
            remaining_chunks = stats['total_chunks'] - stats['completed_chunks']
            log_message(f"  \u26a0️ {remaining_chunks:,} chunks remain unprocessed", log_level="WARNING")
        if has_errors:
            log_message(f"  \u26a0️ {error_stats['total_errors']:,} errors encountered during processing", log_level="WARNING")
        log_message("  \u2139️ Use the same checkpoint file to continue processing", log_level="WARNING")
    
    log_message("\n" + "=" * 80, log_level="WARNING")


##############################################################################
# Main directory processing function
##############################################################################

def process_directory(input_dir: str, output_file: str, chunks_dir: str, model_name: str, 
                     chunk_size: int, question_type: QuestionType, num_answers: int, min_score: int,
                     checkpoint_file: str, force_restart: bool = False,
                     recursive: bool = False, chunks_only: bool = False, workers: int = None):
    """
    Main function for the V16 version with parallel processing:
    1) Create a map of all input files with unique identifiers
    2) Extract and write chunks to files
    3) Process chunks in parallel using a worker pool
    4) Write out question data immediately after each chunk is processed
    5) Provide progress tracking and completion estimation
    """
    global _file_map, _chunk_map, _total_files, _processed_files, _total_chunks
    global _processed_chunks, _extracted_chunks, _start_time, _max_workers, _error_log_file
    
    # Set the max workers
    _max_workers = workers
    
    # Reset start time for accurate timing
    _start_time = time.time()
    
    # Initialize the error log file
    # Create error log filename based on output file
    output_basename = os.path.basename(output_file)
    output_dir = os.path.dirname(output_file)
    output_name = os.path.splitext(output_basename)[0]  # Remove extension
    _error_log_file = os.path.join(output_dir, f"{output_name}_errors.log")
    
    # Log start of processing and clear the error log file
    with open(_error_log_file, 'w', encoding='utf-8') as f:
        f.write(f"Error log for processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Error threshold set to: {_max_error_threshold} (processing will stop if any error type exceeds this count)\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Checkpoint file: {checkpoint_file}\n")
        f.write("=" * 80 + "\n\n")
    
    log_message(f"Error log will be written to: {_error_log_file}")
    
    # Initialize the terminal UI
    ui_initialized = init_terminal_ui()
    if ui_initialized:
        log_message("Terminal UI initialized successfully")
        # Set the question type for display purposes
        if terminal_ui:
            terminal_ui.set_question_type(question_type)
    else:
        log_message("Terminal UI initialization failed, falling back to standard output", log_level="WARNING")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # kill command
    
    # Initialize the checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_file, force_restart)
    # Set the question type for proper display messages
    checkpoint_manager.set_question_type(question_type)
    
    # Save initial state to ensure the checkpoint file exists
    checkpoint_manager.force_save()
    
    try:
        # Start with initializing status
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(
                status_message="Initializing...",
                current_file=input_dir,
                max_workers=_max_workers if _max_workers else max(1, multiprocessing.cpu_count() - 1),
                model_name=model_name
            )
            
        # STEP 1: Map input files and create unique identifiers
        log_message(f"Step 1: Mapping input files in {input_dir}")
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(status_message=f"Mapping input files...")
        
        start_time = time.time()
        
        # Load already processed files from checkpoint
        processed_files = checkpoint_manager.get_processed_files()
        processed_chunks = checkpoint_manager.get_processed_chunks()
        
        # Load error statistics from checkpoint
        error_stats = checkpoint_manager.get_error_stats()
        
        # Update the UI with error stats from checkpoint if available
        if terminal_ui and _use_split_screen:
            # Update all the error statistics
            terminal_ui.update_stats(
                error_file_processing=error_stats.get('error_file_processing', 0),
                error_chunk_extraction=error_stats.get('error_chunk_extraction', 0),
                error_chunk_reading=error_stats.get('error_chunk_reading', 0),
                error_summarizing=error_stats.get('error_summarizing', 0),
                error_question_gen=error_stats.get('error_question_gen', 0),
                error_question_eval=error_stats.get('error_question_eval', 0),
                error_api=error_stats.get('error_api', 0),
                error_other=error_stats.get('error_other', 0),
                low_score_questions=error_stats.get('low_score_questions', 0),
                total_errors=error_stats.get('total_errors', 0)
            )
            
        # Load counter statistics from checkpoint
        counter_stats = checkpoint_manager.get_counter_stats()
        
        # Update global counters from checkpoint atomically
        with _counter_lock:
            # Initialize counters with values from checkpoint
            _extracted_chunks = counter_stats.get('extracted_chunks', 0)
            _processed_chunks = counter_stats.get('processed_chunks', 0)
            
            # We'll set _total_chunks later after we've analyzed all files
            
        # Update global chunk map from checkpoint
        for chunk_id, chunk_data in processed_chunks.items():
            _chunk_map[chunk_id] = chunk_data
        
        # Initialize file map
        if recursive:
            log_message("Recursively searching for files...")
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(status_message="Recursively searching for files...")
            
            file_tuples = find_files_recursively(input_dir, ['.pdf', '.txt', '.md', '.mmd'])
            
            # Create file map
            for rel_path, abs_path in file_tuples:
                file_id = generate_file_id(abs_path)
                # Skip if this file is already fully processed
                if file_id in processed_files and processed_files[file_id].get('status') == 'chunked':
                    log_message(f"File {os.path.basename(abs_path)} already processed, loading from checkpoint")
                    _file_map[file_id] = processed_files[file_id]
                    
                    # Increment counter in a thread-safe way
                    with _counter_lock:
                        _processed_files += 1
                    
                    # Update the UI to show the correct file count
                    if terminal_ui and _use_split_screen:
                        # Force immediate stats update to ensure UI is in sync with counters
                        terminal_ui.update_stats(files_processed=_processed_files)
                        update_global_stats()
                        
                    continue
                    
                _file_map[file_id] = {
                    'file_path': abs_path,
                    'relative_path': rel_path,
                    'filename': os.path.basename(abs_path),
                    'size': os.path.getsize(abs_path),
                    'last_modified': os.path.getmtime(abs_path),
                    'type': os.path.splitext(abs_path)[1].lower()[1:],
                    'status': 'pending',
                    'discovery_time': time.time()
                }
        else:
            # Only look at files in the input directory
            for filename in os.listdir(input_dir):
                if filename.lower().endswith(('.pdf', '.txt', '.md', '.mmd')):
                    file_path = os.path.join(input_dir, filename)
                    file_id = generate_file_id(file_path)
                    
                    # Skip if this file is already fully processed
                    if file_id in processed_files and processed_files[file_id].get('status') == 'chunked':
                        log_message(f"File {filename} already processed, loading from checkpoint")
                        _file_map[file_id] = processed_files[file_id]
                        
                        # Increment counter in a thread-safe way
                        with _counter_lock:
                            _processed_files += 1
                        
                        # Update the UI to show the correct file count
                        if terminal_ui and _use_split_screen:
                            # Force immediate stats update to ensure UI is in sync with counters
                            terminal_ui.update_stats(files_processed=_processed_files)
                            update_global_stats()
                            
                        continue
                        
                    _file_map[file_id] = {
                        'file_path': file_path,
                        'relative_path': filename,
                        'filename': filename,
                        'size': os.path.getsize(file_path),
                        'last_modified': os.path.getmtime(file_path),
                        'type': os.path.splitext(filename)[1].lower()[1:],
                        'status': 'pending',
                        'discovery_time': time.time()
                    }
        
        # Get file counters from checkpoint
        file_count_from_checkpoint = counter_stats.get('total_files', 0)
        processed_count_from_checkpoint = counter_stats.get('processed_files', 0)
        
        # Calculate current files in the map
        current_map_files = len(_file_map)
        
        # Determine the total files, prioritizing the checkpoint value if it's higher
        # This ensures we don't lose count of files that might not be in the current directory 
        _total_files = max(file_count_from_checkpoint, current_map_files)
        
        # Initialize processed files from checkpoint
        _processed_files = processed_count_from_checkpoint
        
        # Update UI with initial file counts immediately
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(
                total_files=_total_files,
                files_processed=_processed_files
            )
        
        # Count files that are fully processed (chunked)
        # First count files marked as processed in current file map
        processed_count = 0  # Track count locally first
        for file_id, file_info in _file_map.items():
            if file_info.get('status') == 'chunked':
                processed_count += 1
        
        # Update counter atomically
        with _counter_lock:
            _processed_files += processed_count
                
        # Then add files from checkpoint that aren't in current map
        # but were processed in a previous run
        checkpoint_count = 0  # Track checkpoint files count locally first
        for file_id, file_info in processed_files.items():
            if file_id not in _file_map and file_info.get('status') == 'chunked':
                checkpoint_count += 1
        
        # Update counter atomically
        with _counter_lock:
            _processed_files += checkpoint_count
                
        # Update the UI with the final counts and recalculate completion percentage
        if terminal_ui and _use_split_screen:
            # Calculate progress percentage for UI display
            file_progress = min(100.0, (_processed_files / max(1, _total_files)) * 100)
            
            # Update the UI with all counts
            terminal_ui.update_stats(
                files_processed=_processed_files,
                total_files=_total_files,
                completion_percentage=file_progress
            )
            # Force an update_global_stats call to ensure consistent display
            update_global_stats()
        
        # Count files that still need processing (in current map but not chunked)
        unprocessed_files = sum(1 for file_info in _file_map.values() if file_info.get('status') != 'chunked')
        mapping_time = time.time() - start_time
        
        # Calculate total files that need processing (includes undetected files from previous runs)
        total_unprocessed = _total_files - _processed_files
        
        log_message(f"Found {_total_files} files total ({_processed_files} already processed, {total_unprocessed} need processing) in {human_readable_time(mapping_time)}")
        
        # Update global stats in UI with consistent values
        if terminal_ui and _use_split_screen:
            # Calculate progress percentage for UI display again to ensure consistency
            file_progress = min(100.0, (_processed_files / max(1, _total_files)) * 100)
            
            terminal_ui.update_stats(
                total_files=_total_files,
                files_processed=_processed_files,
                completion_percentage=file_progress,
                status_message=f"Found {_total_files} files, {total_unprocessed} need processing"
            )
            # Make sure global stats and UI are in sync
            update_global_stats()
        
        # Filter file map to only include unprocessed files
        files_to_process = {
            file_id: file_info for file_id, file_info in _file_map.items() 
            if file_info.get('status') != 'chunked'
        }
        
        # Check if there are any unprocessed files (using our calculated total)
        if _processed_files == _total_files:
            log_message("All files already processed according to checkpoint.")
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(status_message="All files already processed, loading chunks from checkpoint")
                
            # Rebuild file_to_chunks from checkpoint
            file_to_chunks = {}
            for chunk_id, chunk_data in processed_chunks.items():
                file_id = chunk_data.get('file_id')
                if file_id:
                    if file_id not in file_to_chunks:
                        file_to_chunks[file_id] = []
                    file_to_chunks[file_id].append(chunk_id)
        else:
            # STEP 2: Extract and write chunks for files that need processing
            log_message(f"\nStep 2: Extracting and writing chunks to {chunks_dir}")
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(status_message=f"Extracting and writing chunks to {chunks_dir}")
                
            chunk_extraction_start = time.time()
            
            # Make sure the chunks directory exists
            ensure_dir_exists(chunks_dir)
            
            # Create a file_to_chunks mapping - process one file at a time
            file_to_chunks = extract_chunks_sequentially(
                files_to_process, chunks_dir, chunk_size, checkpoint_manager
            )
            
            chunk_extraction_time = time.time() - chunk_extraction_start
            
            log_message(
                f"Extracted {_extracted_chunks} chunks from {_processed_files}/{len(files_to_process)} files "
                f"in {human_readable_time(chunk_extraction_time)}"
            )
            
            # Update UI with extraction statistics
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(
                    chunks_extracted=_extracted_chunks,
                    files_processed=_processed_files,
                    status_message=f"Completed chunk extraction"
                )
                update_global_stats()
            
            # Update checkpoint with extracted chunks
            for file_id, chunk_ids in file_to_chunks.items():
                if file_id in _file_map:
                    checkpoint_manager.add_processed_file(file_id, _file_map[file_id], chunk_ids)
            
            # Merge with already processed chunks from checkpoint
            for file_id, file_info in processed_files.items():
                if file_id not in file_to_chunks and file_id in _file_map:
                    # Find chunks for this file
                    file_chunks = [chunk_id for chunk_id, chunk_data in processed_chunks.items() 
                                  if chunk_data.get('file_id') == file_id]
                    if file_chunks:
                        file_to_chunks[file_id] = file_chunks
        
        # If chunks_only mode, exit after extracting chunks
        if chunks_only:
            log_message("\nChunks extraction complete. Exiting as requested (chunks-only mode).")
            if terminal_ui and _use_split_screen:
                terminal_ui.update_stats(status_message="Completed - chunks-only mode")
            cleanup_ui()
            return
        
        # STEP 3: Process chunks with parallel workers
        log_message(f"\nStep 3: Processing chunks with parallel workers")
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(status_message="Starting chunk processing")
            
        processing_start_time = time.time()
        
        # Get all chunk IDs for processing
        all_chunk_ids = []
        for chunk_ids in file_to_chunks.values():
            all_chunk_ids.extend(chunk_ids)
        
        _total_chunks = len(all_chunk_ids)
        
        # Ensure counters exist before updating the checkpoint
        checkpoint_manager._ensure_counters_exist()
        
        # Update the total chunks count in the checkpoint
        checkpoint_manager.checkpoint_data['counters']['total_chunks'] = _total_chunks
        checkpoint_manager.save()
        
        # Check how many chunks are already processed
        already_processed_chunks = sum(1 for chunk_id in all_chunk_ids if checkpoint_manager.is_chunk_processed(chunk_id))
        _processed_chunks = already_processed_chunks  # Update global counter
        
        # Also update the processed chunks count in the checkpoint
        checkpoint_manager._ensure_counters_exist()
        checkpoint_manager.checkpoint_data['counters']['processed_chunks'] = already_processed_chunks
        checkpoint_manager.save()
        
        log_message(f"Found {len(all_chunk_ids)} total chunks, {already_processed_chunks} already processed")
        
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(
                total_chunks=_total_chunks,
                chunks_processed=already_processed_chunks,
                status_message=f"Processing chunks ({already_processed_chunks}/{_total_chunks} already done)"
            )
            update_global_stats()
        
        # Initialize the worker pool
        init_worker_pool(_max_workers)
                
        # Process all chunks in parallel 
        results = process_chunks_with_parallel_workers(
            all_chunk_ids, chunks_dir, model_name, question_type,
            num_answers, min_score, checkpoint_manager, output_file
        )
        
        # Shutdown the worker pool after processing
        shutdown_worker_pool()
        
        processing_time = time.time() - processing_start_time
        total_time = time.time() - start_time
        
        # Final statistics and summary
        stats = checkpoint_manager.get_completion_stats()
        
        log_message("\nProcessing complete!")
        log_message(f"Total time: {human_readable_time(total_time)}")
        # Ensure the percentage doesn't exceed 100%
        completion_percentage = min(100.0, (_processed_files / max(1, _total_files)) * 100)
        log_message(f"Files processed: {_processed_files}/{_total_files} ({completion_percentage:.1f}%)")
        
        # Ensure chunk percentage doesn't exceed 100%
        chunk_percentage = min(100.0, stats['completion_percentage'])
        log_message(f"Chunks processed: {stats['completed_chunks']}/{stats['total_chunks']} "
              f"({chunk_percentage:.1f}%)")
        if question_type == QuestionType.REASONING_TRACE:
            log_message(f"Traces generated: {checkpoint_manager.get_questions_count()}")
            status_label = "traces"
        else:
            log_message(f"Questions generated: {checkpoint_manager.get_questions_count()}")
            status_label = "questions"
        log_message(f"Results saved to: {output_file}")
        
        # Update UI with final status
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(
                status_message=f"Complete! Saved {checkpoint_manager.get_questions_count()} {status_label} to {output_file}"
            )
            update_global_stats()
        
    except Exception as e:
        log_message(f"Error in main process: {e}", log_level="ERROR")
        import traceback
        trace_str = traceback.format_exc()
        for line in trace_str.split('\n'):
            log_message(line, log_level="ERROR")
            
        # Try to save checkpoint on error
        checkpoint_manager.force_save()
        
        # Update UI with error
        if terminal_ui and _use_split_screen:
            terminal_ui.update_stats(
                status_message=f"Error: {str(e)}"
            )
        
        # Shutdown worker pool if it exists
        if _worker_pool:
            shutdown_worker_pool()
    
    finally:
        # Final report before exiting
        log_message("\n" + "=" * 70, log_level="WARNING")
        log_message("Completing final operations and shutting down...", log_level="WARNING")
        log_message("=" * 70, log_level="WARNING")
        
        # Save checkpoint before exiting
        try:
            log_message("\n- Saving final checkpoint...")
            checkpoint_manager.force_save()
            
            # Print checkpoint statistics
            stats = checkpoint_manager.get_completion_stats()
            error_stats = checkpoint_manager.get_error_stats()
            
            log_message("\nFinal Checkpoint Status:")
            # Ensure the percentage doesn't exceed 100%
            completion_percentage = min(100.0, (_processed_files / max(1, _total_files)) * 100)
            log_message(f"- Files processed: {_processed_files}/{_total_files} ({completion_percentage:.1f}%)")
            log_message(f"- Chunks extracted: {_extracted_chunks}")
            log_message(f"- Chunks processed: {stats['completed_chunks']}/{stats['total_chunks']}")
            
            # Ensure chunk percentage doesn't exceed 100%
            chunk_percentage = min(100.0, stats['completion_percentage'])
            log_message(f"- Completion: {chunk_percentage:.1f}%")
            if question_type == QuestionType.REASONING_TRACE:
                log_message(f"- Traces generated: {checkpoint_manager.get_questions_count()}")
            else:
                log_message(f"- Questions generated: {checkpoint_manager.get_questions_count()}")
            log_message(f"- Total errors: {error_stats['total_errors']}")
            
            # Write final error summary to the error log file
            if _error_log_file:
                try:
                    with open(_error_log_file, 'a', encoding='utf-8') as f:
                        f.write("\n\n" + "=" * 80 + "\n")
                        f.write(f"ERROR SUMMARY (Processing completed at {time.strftime('%Y-%m-%d %H:%M:%S')})\n")
                        f.write("=" * 80 + "\n\n")
                        
                        # Write overall processing statistics
                        f.write(f"Files processed: {_processed_files}/{_total_files} ({completion_percentage:.1f}%)\n")
                        f.write(f"Chunks extracted: {_extracted_chunks}\n")
                        f.write(f"Chunks processed: {stats['completed_chunks']}/{stats['total_chunks']} ({chunk_percentage:.1f}%)\n")
                        if question_type == QuestionType.REASONING_TRACE:
                            f.write(f"Traces generated: {checkpoint_manager.get_questions_count()}\n\n")
                        else:
                            f.write(f"Questions generated: {checkpoint_manager.get_questions_count()}\n\n")
                        
                        # Write error statistics
                        f.write("ERROR COUNTS BY TYPE:\n")
                        f.write(f"- Total errors: {error_stats['total_errors']}\n")
                        f.write(f"- File processing errors: {error_stats['error_file_processing']}\n")
                        f.write(f"- Chunk extraction errors: {error_stats['error_chunk_extraction']}\n")
                        f.write(f"- Chunk reading errors: {error_stats['error_chunk_reading']}\n")
                        f.write(f"- Summarizing errors: {error_stats['error_summarizing']}\n")
                        f.write(f"- Question generation errors: {error_stats['error_question_gen']}\n")
                        f.write(f"- Question evaluation errors: {error_stats['error_question_eval']}\n")
                        f.write(f"- API errors: {error_stats['error_api']}\n")
                        f.write(f"- Low score questions: {error_stats['low_score_questions']}\n")
                        f.write(f"- Other errors: {error_stats['error_other']}\n\n")
                        
                        # Write completion status
                        completion_reason = "normal completion"
                        if _exit_requested:
                            completion_reason = "user interrupt or error threshold exceeded"
                        f.write(f"Process ended due to: {completion_reason}\n")
                        f.write(f"Total processing time: {human_readable_time(time.time() - _start_time)}\n")
                        
                    log_message(f"- Error summary written to {_error_log_file}")
                except Exception as e:
                    log_message(f"- Error writing final error summary: {e}", log_level="ERROR")
                    
            # Display comprehensive exit summary to console
            display_exit_summary(stats, error_stats, completion_percentage, chunk_percentage, checkpoint_manager)
            
        except Exception as e:
            log_message(f"- Error saving final checkpoint: {e}", log_level="ERROR")
        
        # Final cleanup of worker pool if it exists
        if _worker_pool:
            shutdown_worker_pool()
        
        log_message("\n" + "=" * 70, log_level="WARNING")
        log_message("Shutdown complete. You can restart with the same checkpoint file to continue.", log_level="WARNING")
        log_message("=" * 70, log_level="WARNING")
        
        # Final cleanup of UI
        cleanup_ui()


##############################################################################
# Command-line argument parsing
##############################################################################
def write_chunk_to_file(chunk_id: str, chunk_text: str, chunks_dir: str) -> str:
    """
    Write a chunk to a file in the chunks directory.
    Returns the path to the written file.
    """
    # Create a structure like chunks_dir/ab/abcdef1234567890_0001.txt
    # where 'ab' is the first 2 chars of the file_id
    subdir = chunk_id[:2]
    dir_path = os.path.join(chunks_dir, subdir)
    ensure_dir_exists(dir_path)
    
    # Create the file path
    chunk_file_path = os.path.join(dir_path, f"{chunk_id}.txt")
    
    try:
        # Write the chunk to the file
        with open(chunk_file_path, 'w', encoding='utf-8') as f:
            f.write(chunk_text)
        return chunk_file_path
    except PermissionError as e:
        log_message(f"Permission denied writing chunk {chunk_id} to {chunk_file_path}: {e}", 
                   log_level="ERROR", error_type="file_processing")
        return None
    except OSError as e:
        log_message(f"OS error writing chunk {chunk_id} to {chunk_file_path} (disk full/path too long?): {e}", 
                   log_level="ERROR", error_type="file_processing")
        return None
    except UnicodeEncodeError as e:
        log_message(f"Unicode encoding error writing chunk {chunk_id} to {chunk_file_path}: {e}", 
                   log_level="ERROR", error_type="file_processing")
        return None
    except Exception as e:
        log_message(f"Unexpected error writing chunk {chunk_id} to {chunk_file_path}: {e}", 
                   log_level="ERROR", error_type="file_processing")
        return None
        
        
def create_chunk_id(file_id: str, chunk_index: int) -> str:
    """
    Create a unique identifier for a chunk based on the file ID and chunk index.
    """
    return f"{file_id}_{chunk_index:04d}"
    
    
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """
    text = ""
    try:
        # Open the PDF file
        with open(file_path, 'rb') as f:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(f)
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # Only add non-empty text
                if page_text and page_text.strip():
                    text += page_text + "\n\n"
    except FileNotFoundError as e:
        log_message(f"PDF file not found: {file_path}. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
    except PermissionError as e:
        log_message(f"Permission denied accessing PDF file: {file_path}. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
    except PyPDF2.errors.PdfReadError as e:
        log_message(f"PDF read error - file may be corrupted or encrypted: {file_path}. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
    except OSError as e:
        log_message(f"OS error accessing PDF file: {file_path}. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
    except Exception as e:
        log_message(f"Unexpected error extracting text from PDF {file_path}: {e}", 
                   log_level="ERROR", error_type="file_processing")
    
    return text


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a TXT or MD file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        return text
    except FileNotFoundError as e:
        log_message(f"Text file not found: {file_path}. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
        return ""
    except PermissionError as e:
        log_message(f"Permission denied accessing text file: {file_path}. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
        return ""
    except UnicodeDecodeError as e:
        log_message(f"Unicode decode error in text file: {file_path}. Trying fallback encoding. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
        try:
            # Try a different encoding if utf-8 fails
            with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                text = f.read()
            log_message(f"Successfully read {file_path} with latin-1 encoding", log_level="INFO")
            return text
        except Exception as fallback_e:
            log_message(f"Error with fallback encoding for {file_path}: {fallback_e}", 
                       log_level="ERROR", error_type="file_processing")
            return ""
    except OSError as e:
        log_message(f"OS error accessing text file: {file_path}. Error: {e}", 
                   log_level="ERROR", error_type="file_processing")
        return ""
    except Exception as e:
        log_message(f"Unexpected error extracting text from {file_path}: {e}", 
                   log_level="ERROR", error_type="file_processing")
        return ""
            
            
def split_text_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks of approximately chunk_size words each.
    """
    # Normalize whitespace and clean up the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into words
    words = text.split()
    
    # Handle empty text
    if not words:
        return []
    
    # Create chunks
    chunks = []
    chunk_words = []
    
    for word in words:
        chunk_words.append(word)
        
        # When chunk reaches target size, save it and start a new one
        if len(chunk_words) >= chunk_size:
            chunks.append(" ".join(chunk_words))
            chunk_words = []
    
    # Add any remaining words as the final chunk
    if chunk_words:
        chunks.append(" ".join(chunk_words))
    
    return chunks

            
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate questions from text documents. V22 supports multiple-choice questions, free-form questions, and reasoning traces.')
    parser.add_argument('input_directory', help='Directory containing PDF, TXT, or MD files')
    parser.add_argument('--type', type=QuestionType, choices=list(QuestionType), default=QuestionType.MULTIPLE_CHOICE, 
                        help='Type of questions to generate: mc (multiple-choice), qa (free-form), or rt (reasoning-trace)')
    parser.add_argument('--output', default='output.json', help='Output JSON file (default: output.json)')
    parser.add_argument('--chunks-dir', default='chunks', 
                        help='Directory to store extracted chunks. If using default, the name will be based on output file name (e.g., output_name_chunks)')
    parser.add_argument('--model', default='llama', help='Model shortname from model configuration file to use')
    parser.add_argument('--config', default='model_servers.yaml', 
                       help='Path to model configuration file (default: model_servers.yaml)')
    parser.add_argument('--chunk-size', type=int, default=500, help='Approximate number of words per chunk (default: 500)')
    parser.add_argument('--num-answers', type=int, default=7, help='Number of answer choices for each multiple-choice question (default: 7)')
    parser.add_argument('--min-score', type=int, default=7, help='Minimum score (1-10) for keeping a question-answer pair (default: 7)')
    parser.add_argument('--checkpoint', default='checkpoint.json', 
                       help='Checkpoint file to track progress. If using default, the name will be based on output file name (e.g., output_name_checkpoint.json)')
    parser.add_argument('--force-restart', action='store_true', help='Force restart processing from beginning, ignoring checkpoint')
    parser.add_argument('--recursive', action='store_true', help='Recursively search subdirectories for files to process')
    parser.add_argument('--chunks-only', action='store_true', help='Only extract chunks, do not generate questions')
    parser.add_argument('--no-split-screen', action='store_true', help='Disable the split-screen UI and use standard console output')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes to use for parallel processing (default: CPU count - 1)')
    parser.add_argument('--error-threshold', type=int, default=200, help='Maximum number of errors of any type before stopping (default: 200)')
    return parser.parse_args()
    
    


def configure_apis(model_name: str, config_file: str = "model_servers.yaml") -> str:
    """
    Configure the necessary APIs based on model selection.
    
    Args:
        model_name: The model shortname to use
        config_file: Path to the model configuration file
    
    Returns:
        The actual model name to use with the API
    """
    # Batch API support removed in v19
    
    # Load the servers configuration
    try:
        with open(config_file, "r") as f:
            servers_config = yaml.safe_load(f)
    except Exception as e:
        log_message(f"Error loading {config_file}: {e}", log_level="ERROR")
        sys.exit(1)
    
    # Find the selected model's configuration
    selected_server = None
    for server in servers_config["servers"]:
        if server["shortname"] == model_name:
            selected_server = server
            break
    
    if not selected_server:
        log_message(f"Error: Model '{model_name}' not found in {config_file}", log_level="ERROR")
        log_message(f"Available models: {', '.join(s['shortname'] for s in servers_config['servers'])}", log_level="INFO")
        sys.exit(1)
    
    # Configure OpenAI API with server details
    # Assume all servers are OpenAI compatible without checking type
    api_key = selected_server.get("openai_api_key", "dummy_key_not_used")
    # Handle environment variables in the API key
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var = api_key[2:-1]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            log_message(f"Error: Environment variable {env_var} not set", log_level="ERROR")
            sys.exit(1)
    
    # Initialize the OpenAI client
    global _openai_client
    
    # Prepare client configuration
    client_config = {
        "api_key": api_key,
    }
    
    # Set base URL if provided
    if "openai_api_base" in selected_server:
        client_config["base_url"] = selected_server.get("openai_api_base")
    
    # Set organization if provided
    if "org_id" in selected_server:
        client_config["organization"] = selected_server["org_id"]
    
    # Create the client
    _openai_client = OpenAI(**client_config)
    
    # Get the actual model name to use
    actual_model_name = selected_server.get("openai_model")
    
    base_url = selected_server.get("openai_api_base", "https://api.openai.com/v1")
    log_message(f"Configured OpenAI API with base URL: {base_url}")
    log_message(f"Using model shortname: {model_name}")
    log_message(f"Actual model identifier: {actual_model_name}")
    
    # Batch API support removed in v19
    
    return actual_model_name


def main():
    """Main entry point function."""
    global _use_split_screen, _max_error_threshold
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set error threshold from command line argument
    _max_error_threshold = args.error_threshold
    log_message(f"Error threshold set to {_max_error_threshold}", log_level="INFO")
    
    # Configure split-screen UI mode
    _use_split_screen = not args.no_split_screen
    
    # Configure the OpenAI API for the selected model
    # This returns the actual model name to use with the API
    actual_model_name = configure_apis(args.model, args.config)
    
    # Batch API support removed in v19
    
    # Determine checkpoint file name and chunks directory based on output file 
    # if the defaults are used to avoid conflicts between concurrent jobs
    checkpoint_file = args.checkpoint
    chunks_dir = args.chunks_dir
    
    # Only customize paths if using custom output file
    if args.output != 'output.json':
        # Extract base name from output file without extension
        output_base = os.path.splitext(args.output)[0]
        
        # Create custom checkpoint file name if using default
        if checkpoint_file == 'checkpoint.json':
            checkpoint_file = f"{output_base}_checkpoint.json"
            log_message(f"Using checkpoint file: {checkpoint_file}")
        
        # Create custom chunks directory if using default
        if chunks_dir == 'chunks':
            chunks_dir = f"{output_base}_chunks"
            log_message(f"Using chunks directory: {chunks_dir}")
    
    # Process the input directory
    process_directory(
        input_dir=args.input_directory,
        output_file=args.output,
        chunks_dir=chunks_dir,
        model_name=actual_model_name,  # Use the actual model name here
        chunk_size=args.chunk_size,
        question_type=args.type,
        num_answers=args.num_answers,
        min_score=args.min_score,
        checkpoint_file=checkpoint_file,
        force_restart=args.force_restart,
        recursive=args.recursive,
        chunks_only=args.chunks_only,
        workers=args.workers
    )


if __name__ == "__main__":
    main()