import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))

try:
    from app import app
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

def run_tests():
    print("Initializing Flask Local Test Client...")
    
    # We use Flask's built-in test client so we don't even need the server running!
    # It communicates directly with the API logic.
    with app.test_client() as client:
        success_count = 0
        failure_list = []
        
        print("Testing INT001 through INT100...")
        for i in range(1, 101):
            intern_id = f"INT{i:03d}"
            
            response = client.get(f"/predict-by-id?id={intern_id}")
            
            if response.status_code == 200:
                data = response.get_json()
                # Verify that it isn't an error response wrapped in a 200
                if "error" in data:
                    failure_list.append((intern_id, data["error"]))
                else:
                    success_count += 1
            else:
                failure_list.append((intern_id, f"HTTP {response.status_code}"))
                
        print("\n" + "="*50)
        print("API VERIFICATION RESULTS")
        print("="*50)
        print(f"Total Tested: 100")
        print(f"Successful:   {success_count}")
        print(f"Failed:       {len(failure_list)}")
        
        if failure_list:
            print("\nFailures:")
            for fid, err in failure_list:
                print(f"  - {fid}: {err}")
            sys.exit(1)
        else:
            print("\nALL 100 INTERN PROFILES WORKING FLAWLESSLY!")
            sys.exit(0)

if __name__ == "__main__":
    run_tests()
