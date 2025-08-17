#!/usr/bin/env python
"""Test runner for OChem Helper."""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'mcp'))


def run_unit_tests(verbosity=2):
    """Run unit tests."""
    print("=" * 70)
    print("Running Unit Tests")
    print("=" * 70)
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    suite = loader.discover('unit', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests(verbosity=2):
    """Run integration tests."""
    print("\n" + "=" * 70)
    print("Running Integration Tests")
    print("=" * 70)
    
    # Discover and run integration tests
    loader = unittest.TestLoader()
    suite = loader.discover('integration', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_path, verbosity=2):
    """Run a specific test file or test case."""
    print(f"Running specific test: {test_path}")
    
    # Load the specific test
    loader = unittest.TestLoader()
    
    if '::' in test_path:
        # Format: test_file.py::TestClass::test_method
        parts = test_path.split('::')
        module_name = parts[0].replace('.py', '').replace('/', '.')
        
        if len(parts) == 2:
            # Load specific test class
            suite = loader.loadTestsFromName(f"{module_name}.{parts[1]}")
        elif len(parts) == 3:
            # Load specific test method
            suite = loader.loadTestsFromName(f"{module_name}.{parts[1]}.{parts[2]}")
    else:
        # Load entire test file
        suite = loader.discover(os.path.dirname(test_path), 
                              pattern=os.path.basename(test_path))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_coverage_tests():
    """Run tests with coverage reporting."""
    try:
        import coverage
    except ImportError:
        print("Coverage.py not installed. Install with: pip install coverage")
        return False
    
    print("Running tests with coverage...")
    
    # Initialize coverage
    cov = coverage.Coverage(source=['../src', '../mcp'])
    cov.start()
    
    # Run all tests
    unit_success = run_unit_tests(verbosity=1)
    integration_success = run_integration_tests(verbosity=1)
    
    # Stop coverage and generate report
    cov.stop()
    cov.save()
    
    print("\n" + "=" * 70)
    print("Coverage Report")
    print("=" * 70)
    
    cov.report()
    
    # Generate HTML report
    html_dir = Path(__file__).parent / 'htmlcov'
    cov.html_report(directory=str(html_dir))
    print(f"\nDetailed HTML report generated in: {html_dir}")
    
    return unit_success and integration_success


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run OChem Helper tests')
    parser.add_argument('test_path', nargs='?', help='Specific test to run')
    parser.add_argument('-u', '--unit', action='store_true', 
                       help='Run only unit tests')
    parser.add_argument('-i', '--integration', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('-c', '--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('-v', '--verbosity', type=int, default=2,
                       help='Test output verbosity (0-2)')
    parser.add_argument('-f', '--failfast', action='store_true',
                       help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Change to tests directory
    os.chdir(Path(__file__).parent)
    
    # Configure unittest settings
    if args.failfast:
        unittest.TestProgram.failfast = True
    
    success = True
    
    try:
        if args.coverage:
            success = run_coverage_tests()
        elif args.test_path:
            success = run_specific_test(args.test_path, args.verbosity)
        elif args.unit:
            success = run_unit_tests(args.verbosity)
        elif args.integration:
            success = run_integration_tests(args.verbosity)
        else:
            # Run all tests
            unit_success = run_unit_tests(args.verbosity)
            integration_success = run_integration_tests(args.verbosity)
            success = unit_success and integration_success
    
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        success = False
    
    # Print summary
    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()