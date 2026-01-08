import time
import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{command}': {e.stderr}")
        return None

def main():
    print("Starting file watcher... (Press Ctrl+C to stop)")
    print("Monitoring for changes to commit and push to 'main'...")
    
    while True:
        try:
            # Check for changes
            status = run_command("git status --porcelain")
            
            if status:
                print(f"Changes detected:\n{status}")
                
                # Add changes
                print("Adding changes...")
                run_command("git add .")
                
                # Commit
                print("Committing...")
                commit_msg = f"Auto-commit: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                run_command(f'git commit -m "{commit_msg}"')
                
                # Push
                print("Pushing to main...")
                push_result = run_command("git push origin main")
                if push_result is not None:
                     print("Successfully pushed to main.")
                
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nStopping watcher.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
