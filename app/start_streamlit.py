import subprocess
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")
    
    print("🛡️  Starting FinShield Web Interface...")
    print(f"📁 App location: {app_path}")
    print("🌐 Server will start at: http://localhost:8501")
    print("\n⚠️  Make sure the API is running at http://localhost:8000")
    print("   Run: python api\\start_api.py\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            app_path,
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--global.developmentMode=false"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Streamlit application stopped")
    except Exception as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
