import subprocess

def install_dependencies():
    """Install project dependencies."""
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def run_tests():
    """Run project tests."""
    subprocess.run(["pytest"])

def main():
    """Main build script."""
    install_dependencies()
    run_tests()

if __name__ == "__main__":
    main()
