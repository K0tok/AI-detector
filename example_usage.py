#!/usr/bin/env python3
"""
Example usage of the AI Text Detector
This script demonstrates how to use the AITextDetector class in your own code
"""

from ai_detector import AITextDetector

def main():
    # Create an instance of the detector
    detector = AITextDetector()
    
    # Example: Analyze a document
    file_path = "example_russian.docx"  # Use the Russian example file we created
    results = detector.process_docx_file(file_path)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    print("Analysis Results:")
    print(f"File: {results['file_path']}")
    print(f"AI Probability: {results['analysis']['ai_probability']}%")
    print(f"Issues found: {len(results['analysis']['issues'])}")
    print(f"Suggestions: {len(results['suggestions'])}")
    
    # Show some suggestions
    if results['suggestions']:
        print("\nFirst suggestion:")
        print(f"- Issue: {results['suggestions'][0]['issue']}")
        print(f"- Suggestion: {results['suggestions'][0]['suggestion']}")

if __name__ == "__main__":
    main()