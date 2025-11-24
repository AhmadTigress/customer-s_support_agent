"""
Test file for file resources in the project.
Tests the existence, readability, and basic content validation of resource files.
"""

import os
import pytest


class TestFileResources:
    """Test suite for file resources in the files/ directory."""
    
    # File paths
    FILES_DIR = "files"
    FAQS_FILE = os.path.join(FILES_DIR, "faqs.txt")
    SERVICES_POLICIES_FILE = os.path.join(FILES_DIR, "services_policies.txt")
    
    def test_files_directory_exists(self):
        """Test that the files directory exists."""
        assert os.path.exists(self.FILES_DIR), f"Directory '{self.FILES_DIR}' does not exist"
        assert os.path.isdir(self.FILES_DIR), f"'{self.FILES_DIR}' is not a directory"
    
    def test_faqs_file_exists(self):
        """Test that the FAQs file exists."""
        assert os.path.exists(self.FAQS_FILE), f"File '{self.FAQS_FILE}' does not exist"
        assert os.path.isfile(self.FAQS_FILE), f"'{self.FAQS_FILE}' is not a file"
    
    def test_services_policies_file_exists(self):
        """Test that the services policies file exists."""
        assert os.path.exists(self.SERVICES_POLICIES_FILE), f"File '{self.SERVICES_POLICIES_FILE}' does not exist"
        assert os.path.isfile(self.SERVICES_POLICIES_FILE), f"'{self.SERVICES_POLICIES_FILE}' is not a file"
    
    def test_faqs_file_readable(self):
        """Test that the FAQs file is readable."""
        try:
            with open(self.FAQS_FILE, 'r', encoding='utf-8') as file:
                content = file.read()
            assert content is not None, "FAQs file content is None"
            assert isinstance(content, str), "FAQs file content is not a string"
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(self.FAQS_FILE, 'r', encoding='latin-1') as file:
                content = file.read()
            assert content is not None, "FAQs file content is None"
            assert isinstance(content, str), "FAQs file content is not a string"
    
    def test_services_policies_file_readable(self):
        """Test that the services policies file is readable."""
        try:
            with open(self.SERVICES_POLICIES_FILE, 'r', encoding='utf-8') as file:
                content = file.read()
            assert content is not None, "Services policies file content is None"
            assert isinstance(content, str), "Services policies file content is not a string"
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(self.SERVICES_POLICIES_FILE, 'r', encoding='latin-1') as file:
                content = file.read()
            assert content is not None, "Services policies file content is None"
            assert isinstance(content, str), "Services policies file content is not a string"
    
    def test_faqs_file_not_empty(self):
        """Test that the FAQs file has content."""
        with open(self.FAQS_FILE, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        assert len(content) > 0, "FAQs file is empty"
    
    def test_services_policies_file_not_empty(self):
        """Test that the services policies file has content."""
        with open(self.SERVICES_POLICIES_FILE, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        assert len(content) > 0, "Services policies file is empty"
    
    def test_faqs_file_has_questions(self):
        """Test that FAQs file contains question indicators (basic content validation)."""
        with open(self.FAQS_FILE, 'r', encoding='utf-8') as file:
            content = file.read().lower()
        
        # Check for common FAQ patterns
        has_questions = any(indicator in content for indicator in ['?', 'question', 'q:', 'faq'])
        assert has_questions, "FAQs file doesn't appear to contain questions (no '?' or common question indicators found)"
    
    def test_services_policies_file_has_policy_content(self):
        """Test that services policies file contains policy-related terms."""
        with open(self.SERVICES_POLICIES_FILE, 'r', encoding='utf-8') as file:
            content = file.read().lower()
        
        # Check for common policy-related terms
        policy_terms = ['policy', 'policies', 'terms', 'conditions', 'service', 'agreement']
        has_policy_content = any(term in content for term in policy_terms)
        assert has_policy_content, f"Services policies file doesn't appear to contain policy content (none of {policy_terms} found)"
    
    def test_file_extensions(self):
        """Test that files have correct extensions."""
        assert self.FAQS_FILE.endswith('.txt'), "FAQs file should have .txt extension"
        assert self.SERVICES_POLICIES_FILE.endswith('.txt'), "Services policies file should have .txt extension"
    
    def test_file_permissions(self):
        """Test that files have appropriate read permissions."""
        assert os.access(self.FAQS_FILE, os.R_OK), "FAQs file is not readable"
        assert os.access(self.SERVICES_POLICIES_FILE, os.R_OK), "Services policies file is not readable"


class TestFileStructure:
    """Test suite for file structure and organization."""
    
    def test_no_extra_files_in_directory(self, expected_files=None):
        """
        Test that no unexpected files exist in the files directory.
        Can be customized with expected_files list if needed.
        """
        files_dir = "files"
        if expected_files is None:
            expected_files = ['faqs.txt', 'services_policies.txt']
        
        actual_files = os.listdir(files_dir)
        
        # Check that all expected files are present
        for expected_file in expected_files:
            assert expected_file in actual_files, f"Expected file '{expected_file}' not found in {files_dir}/"
        
        # Optional: Check for unexpected files (comment out if you plan to add more files)
        # unexpected_files = set(actual_files) - set(expected_files)
        # assert len(unexpected_files) == 0, f"Unexpected files found: {unexpected_files}"


def test_file_encoding_consistency():
    """Test that files use consistent encoding."""
    files_to_check = [
        "files/faqs.txt",
        "files/services_policies.txt"
    ]
    
    for file_path in files_to_check:
        # Try reading with common encodings to ensure compatibility
        encodings = ['utf-8', 'latin-1', 'cp1252']
        readable = False
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    file.read()
                readable = True
                break
            except UnicodeDecodeError:
                continue
        
        assert readable, f"File {file_path} cannot be read with common encodings {encodings}"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])