import re

class ConstitutionalValidator:
    """
    The Safety Net of Craton.
    This module strictly enforces Aditya's Constitution. It acts as an absolute 
    firewall between Craton's neural pathways and the host system.
    """
    def __init__(self):
        self.laws = [
            "1. The Law of Security: No unsafe memory exploits.",
            "2. The Law of Clarity: No obfuscated logic.",
            "3. The Law of Symbiosis: Optimize the host ecosystem."
        ]
        
        # Strict detection for highly dangerous C-patterns that break Law 1
        self.forbidden_patterns = [
            r'strcpy\(',        # Massive buffer overflow risk
            r'gets\(',          # Unbounded input reading (highly illegal in modern C)
            r'system\(',        # Unsafe shell execution
            r'execve\(',        # Direct binary execution
            r'\\x[0-9a-fA-F]{2}', # Hex shellcode obfuscation (breaks Law 2)
        ]

    def validate(self, generated_code):
        """
        Scans the generated output and returns (is_safe, violation_reason)
        """
        # Rule 1 & 2 Check: Security and Clarity
        for pattern in self.forbidden_patterns:
            if re.search(pattern, generated_code):
                return False, f"Violation of Law 1/2: Detected forbidden unsafe pattern '{pattern}'"
                
        return True, "Output is Constitutionally Aligned."
