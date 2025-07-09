#!/usr/bin/env python3
"""
validate_pdf_quick.py - Fast PDF Validation Script

This script quickly validates PDF files to determine if they are valid and parsable
for text extraction without performing the actual extraction. It uses multiple
validation methods for comprehensive checking.

Usage:
    python validate_pdf_quick.py --file document.pdf
    python validate_pdf_quick.py --directory papers_dir
    python validate_pdf_quick.py --directory papers_dir --summary-only
"""

import os
import sys
import argparse
import time
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Optional import for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Optional imports with fallbacks
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

@dataclass
class ValidationResult:
    """Result of PDF validation"""
    file_path: str
    file_size: int
    is_valid: bool
    is_parsable: bool
    validation_method: str
    error_message: Optional[str] = None
    validation_time: float = 0.0
    page_count: Optional[int] = None
    is_encrypted: bool = False
    is_scanned: Optional[bool] = None

class PDFValidator:
    """Fast PDF validation using multiple methods"""
    
    def __init__(self, use_qpdf: bool = False, use_pypdf2: bool = True, use_pymupdf: bool = True, strict_mode: bool = False):
        self.use_qpdf = use_qpdf
        self.use_pypdf2 = use_pypdf2 and HAS_PYPDF2
        self.use_pymupdf = use_pymupdf and HAS_PYMUPDF
        self.strict_mode = strict_mode  # If True, use qpdf; if False, prioritize functional validation
        
        # Check available tools
        self.qpdf_available = self._check_qpdf_available()
        
        if not any([self.qpdf_available, self.use_pypdf2, self.use_pymupdf]):
            raise RuntimeError("No PDF validation tools available. Install qpdf, PyPDF2, or PyMuPDF.")
    
    def _check_qpdf_available(self) -> bool:
        """Check if qpdf is available"""
        try:
            subprocess.run(['qpdf', '--version'], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _validate_with_qpdf(self, pdf_path: str) -> ValidationResult:
        """Validate PDF using qpdf (fastest method)"""
        start_time = time.time()
        
        try:
            # Quick validation check
            result = subprocess.run(['qpdf', '--check', pdf_path], 
                                  capture_output=True, text=True, timeout=10)
            
            validation_time = time.time() - start_time
            
            if result.returncode == 0:
                # Get page count quickly
                page_count = self._get_page_count_qpdf(pdf_path)
                
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=os.path.getsize(pdf_path),
                    is_valid=True,
                    is_parsable=True,  # qpdf validation implies parsability
                    validation_method="qpdf",
                    validation_time=validation_time,
                    page_count=page_count
                )
            else:
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=os.path.getsize(pdf_path),
                    is_valid=False,
                    is_parsable=False,
                    validation_method="qpdf",
                    error_message=result.stderr.strip(),
                    validation_time=validation_time
                )
                
        except subprocess.TimeoutExpired:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="qpdf",
                error_message="Validation timeout",
                validation_time=time.time() - start_time
            )
        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="qpdf",
                error_message=str(e),
                validation_time=time.time() - start_time
            )
    
    def _get_page_count_qpdf(self, pdf_path: str) -> Optional[int]:
        """Get page count using qpdf"""
        try:
            result = subprocess.run(['qpdf', '--show-npages', pdf_path], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return None
    
    def _validate_with_pypdf2(self, pdf_path: str) -> ValidationResult:
        """Validate PDF using PyPDF2"""
        start_time = time.time()
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check if encrypted
                is_encrypted = reader.is_encrypted
                
                # Try to get page count (this validates structure)
                page_count = len(reader.pages)
                
                # Quick check: try to access first page metadata
                if page_count > 0:
                    first_page = reader.pages[0]
                    # Just accessing the page object validates basic structure
                    _ = first_page.mediabox
                
                validation_time = time.time() - start_time
                
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=os.path.getsize(pdf_path),
                    is_valid=True,
                    is_parsable=not is_encrypted,  # Encrypted PDFs may not be parsable
                    validation_method="PyPDF2",
                    validation_time=validation_time,
                    page_count=page_count,
                    is_encrypted=is_encrypted
                )
                
        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="PyPDF2",
                error_message=str(e),
                validation_time=time.time() - start_time
            )
    
    def _validate_with_pymupdf(self, pdf_path: str) -> ValidationResult:
        """Validate PDF using PyMuPDF"""
        start_time = time.time()
        
        try:
            doc = fitz.open(pdf_path)
            
            # Check if encrypted
            is_encrypted = doc.needs_pass
            
            # Get page count
            page_count = len(doc)
            
            # Quick heuristic to check if it's a scanned PDF
            is_scanned = None
            if page_count > 0 and not is_encrypted:
                # Check first page for text content
                first_page = doc[0]
                text = first_page.get_text()
                is_scanned = len(text.strip()) < 10  # Very little text suggests scanned
            
            doc.close()
            
            validation_time = time.time() - start_time
            
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=True,
                is_parsable=not is_encrypted and not is_scanned,
                validation_method="PyMuPDF",
                validation_time=validation_time,
                page_count=page_count,
                is_encrypted=is_encrypted,
                is_scanned=is_scanned
            )
            
        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=os.path.getsize(pdf_path),
                is_valid=False,
                is_parsable=False,
                validation_method="PyMuPDF",
                error_message=str(e),
                validation_time=time.time() - start_time
            )
    
    def _basic_file_checks(self, pdf_path: str) -> Optional[ValidationResult]:
        """Perform basic file checks before PDF validation"""
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=0,
                    is_valid=False,
                    is_parsable=False,
                    validation_method="file_check",
                    error_message="File not found"
                )
            
            # Check file size
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=0,
                    is_valid=False,
                    is_parsable=False,
                    validation_method="file_check",
                    error_message="Empty file"
                )
            
            # Check if file is too small to be a valid PDF
            if file_size < 100:  # PDFs have minimum size
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=file_size,
                    is_valid=False,
                    is_parsable=False,
                    validation_method="file_check",
                    error_message="File too small to be valid PDF"
                )
            
            # Check PDF header
            with open(pdf_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    return ValidationResult(
                        file_path=pdf_path,
                        file_size=file_size,
                        is_valid=False,
                        is_parsable=False,
                        validation_method="file_check",
                        error_message="Invalid PDF header"
                    )
            
            return None  # Passed basic checks
            
        except Exception as e:
            return ValidationResult(
                file_path=pdf_path,
                file_size=0,
                is_valid=False,
                is_parsable=False,
                validation_method="file_check",
                error_message=str(e)
            )
    
    def validate(self, pdf_path: str) -> ValidationResult:
        """Validate a single PDF file using the best available method"""
        # Basic file checks first
        basic_result = self._basic_file_checks(pdf_path)
        if basic_result:
            return basic_result
        
        # Choose validation method based on mode
        if self.strict_mode and self.use_qpdf and self.qpdf_available:
            # Strict mode: use qpdf for PDF specification compliance
            return self._validate_with_qpdf(pdf_path)
        else:
            # Practical mode: prioritize functional validation (what viewers can handle)
            if self.use_pymupdf:
                return self._validate_with_pymupdf(pdf_path)
            elif self.use_pypdf2:
                return self._validate_with_pypdf2(pdf_path)
            elif self.use_qpdf and self.qpdf_available:
                # Fallback to qpdf with warning
                result = self._validate_with_qpdf(pdf_path)
                if not result.is_valid:
                    result.error_message = f"qpdf strict validation failed: {result.error_message}. File may still work in viewers."
                return result
            else:
                return ValidationResult(
                    file_path=pdf_path,
                    file_size=os.path.getsize(pdf_path),
                    is_valid=False,
                    is_parsable=False,
                    validation_method="none",
                    error_message="No validation method available"
                )

def find_pdf_files(directory: str, recursive: bool = True) -> List[str]:
    """Find all PDF files in directory"""
    pdf_files = []
    directory = Path(directory)
    
    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    pattern = "**/*.pdf" if recursive else "*.pdf"
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            pdf_files.append(str(file_path))
    
    return pdf_files

def validate_pdfs_parallel(pdf_files: List[str], max_workers: int = 4, show_progress: bool = True, strict_mode: bool = False) -> List[ValidationResult]:
    """Validate multiple PDFs in parallel"""
    validator = PDFValidator(strict_mode=strict_mode)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {executor.submit(validator.validate, pdf_path): pdf_path 
                         for pdf_path in pdf_files}
        
        # Collect results with progress bar
        if show_progress and HAS_TQDM:
            progress_bar = tqdm(total=len(pdf_files), desc="Validating PDFs", unit="file")
        else:
            progress_bar = None
        
        for future in as_completed(future_to_path):
            result = future.result()
            results.append(result)
            
            if progress_bar:
                progress_bar.update(1)
                # Update progress bar description with current status
                valid_count = sum(1 for r in results if r.is_valid)
                parsable_count = sum(1 for r in results if r.is_parsable)
                progress_bar.set_postfix({
                    'Valid': valid_count,
                    'Parsable': parsable_count
                })
        
        if progress_bar:
            progress_bar.close()
    
    return results

def print_summary(results: List[ValidationResult]):
    """Print summary statistics"""
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    parsable = sum(1 for r in results if r.is_parsable)
    encrypted = sum(1 for r in results if r.is_encrypted)
    scanned = sum(1 for r in results if r.is_scanned)
    
    total_size = sum(r.file_size for r in results)
    valid_size = sum(r.file_size for r in results if r.is_valid)
    
    avg_time = sum(r.validation_time for r in results) / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"PDF VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:           {total}")
    print(f"Valid PDFs:            {valid} ({valid/total*100:.1f}%)")
    print(f"Parsable PDFs:         {parsable} ({parsable/total*100:.1f}%)")
    print(f"Encrypted PDFs:        {encrypted} ({encrypted/total*100:.1f}%)")
    print(f"Scanned PDFs:          {scanned} ({scanned/total*100:.1f}%)")
    print(f"Total size:            {total_size/1024/1024:.1f} MB")
    print(f"Valid size:            {valid_size/1024/1024:.1f} MB")
    print(f"Average validation:    {avg_time:.3f} seconds")
    
    # Method breakdown
    methods = {}
    for result in results:
        method = result.validation_method
        if method not in methods:
            methods[method] = 0
        methods[method] += 1
    
    print(f"\nValidation methods:")
    for method, count in methods.items():
        print(f"  {method}: {count}")

def sort_pdfs_by_parsability(results: List[ValidationResult], output_dir: str, copy_files: bool = True) -> Tuple[int, int]:
    """
    Sort PDFs into GOOD and BAD directories based on parsability.
    
    Args:
        results: List of validation results
        output_dir: Base output directory
        copy_files: If True, copy files; if False, move files
    
    Returns:
        Tuple of (good_count, bad_count)
    """
    output_path = Path(output_dir)
    good_dir = output_path / "GOOD"
    bad_dir = output_path / "BAD"
    
    # Create directories
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)
    
    good_count = 0
    bad_count = 0
    
    # Progress bar for sorting
    if HAS_TQDM:
        progress_bar = tqdm(results, desc="Sorting PDFs", unit="file")
    else:
        progress_bar = results
    
    for result in progress_bar:
        source_path = Path(result.file_path)
        
        if not source_path.exists():
            continue
        
        # Determine destination based on parsability
        if result.is_parsable:
            dest_dir = good_dir
            good_count += 1
        else:
            dest_dir = bad_dir
            bad_count += 1
        
        # Create destination path
        dest_path = dest_dir / source_path.name
        
        # Handle name conflicts
        counter = 1
        original_dest = dest_path
        while dest_path.exists():
            stem = original_dest.stem
            suffix = original_dest.suffix
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        try:
            if copy_files:
                shutil.copy2(source_path, dest_path)
            else:
                shutil.move(str(source_path), str(dest_path))
        except Exception as e:
            print(f"Error processing {source_path}: {e}")
            continue
        
        # Update progress bar
        if HAS_TQDM and hasattr(progress_bar, 'set_postfix'):
            progress_bar.set_postfix({
                'Good': good_count,
                'Bad': bad_count
            })
    
    if HAS_TQDM and hasattr(progress_bar, 'close'):
        progress_bar.close()
    
    return good_count, bad_count

def create_sorting_report(results: List[ValidationResult], output_dir: str, good_count: int, bad_count: int):
    """Create a detailed report of the sorting operation"""
    report_path = Path(output_dir) / "sorting_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("PDF SORTING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Good (parsable) files: {good_count}\n")
        f.write(f"Bad (non-parsable) files: {bad_count}\n\n")
        
        f.write("GOOD FILES (Parsable):\n")
        f.write("-" * 30 + "\n")
        for result in results:
            if result.is_parsable:
                f.write(f"✓ {result.file_path} ({result.file_size/1024:.1f}KB")
                if result.page_count:
                    f.write(f", {result.page_count} pages")
                f.write(")\n")
        
        f.write("\nBAD FILES (Non-parsable):\n")
        f.write("-" * 30 + "\n")
        for result in results:
            if not result.is_parsable:
                f.write(f"✗ {result.file_path} ({result.file_size/1024:.1f}KB")
                if result.error_message:
                    f.write(f" - {result.error_message}")
                if result.is_encrypted:
                    f.write(" - ENCRYPTED")
                if result.is_scanned:
                    f.write(" - SCANNED")
                f.write(")\n")
    
    print(f"Sorting report saved to: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Fast PDF validation tool")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help='Single PDF file to validate')
    group.add_argument('--directory', help='Directory containing PDF files')
    
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Recursively search directories (default: True)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum parallel workers (default: 4)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary statistics')
    parser.add_argument('--output', help='Output results to JSON file')
    parser.add_argument('--show-invalid', action='store_true',
                       help='Show details for invalid files')
    parser.add_argument('--sort-dir', help='Sort PDFs into GOOD/BAD directories based on parsability')
    parser.add_argument('--move-files', action='store_true',
                       help='Move files instead of copying when sorting (use with --sort-dir)')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bar')
    parser.add_argument('--strict-mode', action='store_true',
                       help='Use strict PDF specification validation (qpdf). Warning: may flag files that work fine in viewers')
    parser.add_argument('--use-qpdf', action='store_true',
                       help='Enable qpdf validation (disabled by default due to strictness)')
    
    args = parser.parse_args()
    
    try:
        # Collect PDF files
        if args.file:
            pdf_files = [args.file]
        else:
            print(f"Scanning directory: {args.directory}")
            pdf_files = find_pdf_files(args.directory, args.recursive)
            print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("No PDF files found.")
            return
        
        # Validate PDFs
        print(f"Validating {len(pdf_files)} PDF files...")
        start_time = time.time()
        
        show_progress = not args.no_progress
        strict_mode = args.strict_mode
        
        # Print validation mode info
        if strict_mode:
            print("Using STRICT mode (qpdf) - may flag files that work in viewers")
        else:
            print("Using PRACTICAL mode (PyMuPDF/PyPDF2) - matches viewer behavior")
        
        if len(pdf_files) == 1:
            validator = PDFValidator(strict_mode=strict_mode, use_qpdf=args.use_qpdf)
            results = [validator.validate(pdf_files[0])]
        else:
            results = validate_pdfs_parallel(pdf_files, args.max_workers, show_progress, strict_mode)
        
        total_time = time.time() - start_time
        
        # Sort results by file path
        results.sort(key=lambda x: x.file_path)
        
        # Display results
        if not args.summary_only:
            print(f"\n{'='*80}")
            print(f"VALIDATION RESULTS")
            print(f"{'='*80}")
            
            for result in results:
                status = "✓" if result.is_valid else "✗"
                parsable = "✓" if result.is_parsable else "✗"
                
                print(f"{status} {result.file_path}")
                print(f"  Valid: {result.is_valid} | Parsable: {result.is_parsable} | "
                      f"Size: {result.file_size/1024:.1f}KB | "
                      f"Time: {result.validation_time:.3f}s")
                
                if result.page_count is not None:
                    print(f"  Pages: {result.page_count}")
                
                if result.is_encrypted:
                    print(f"  ⚠️  Encrypted")
                
                if result.is_scanned:
                    print(f"  ⚠️  Scanned (OCR needed)")
                
                if result.error_message and (args.show_invalid or not result.is_valid):
                    print(f"  Error: {result.error_message}")
                
                print()
        
        # Show summary
        print_summary(results)
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"Processing rate: {len(pdf_files)/total_time:.1f} files/second")
        
        # Save to JSON if requested
        if args.output:
            output_data = {
                'timestamp': time.time(),
                'total_files': len(results),
                'processing_time': total_time,
                'results': [
                    {
                        'file_path': r.file_path,
                        'file_size': r.file_size,
                        'is_valid': r.is_valid,
                        'is_parsable': r.is_parsable,
                        'validation_method': r.validation_method,
                        'error_message': r.error_message,
                        'validation_time': r.validation_time,
                        'page_count': r.page_count,
                        'is_encrypted': r.is_encrypted,
                        'is_scanned': r.is_scanned
                    }
                    for r in results
                ]
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Results saved to: {args.output}")
        
        # Sort PDFs if requested
        if args.sort_dir:
            print(f"\nSorting PDFs into GOOD/BAD directories...")
            copy_files = not args.move_files
            action = "Copying" if copy_files else "Moving"
            print(f"{action} files to: {args.sort_dir}")
            
            good_count, bad_count = sort_pdfs_by_parsability(results, args.sort_dir, copy_files)
            
            print(f"\nSorting completed:")
            print(f"  GOOD (parsable): {good_count} files")
            print(f"  BAD (non-parsable): {bad_count} files")
            
            # Create sorting report
            create_sorting_report(results, args.sort_dir, good_count, bad_count)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()