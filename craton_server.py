from flask import Flask, render_template, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    # Serves the massive glassmorphism UI
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    # TERMINAL HOOK: If the message starts with '/', Craton executes it directly on your Android tablet!
    if user_message.startswith('/'):
        cmd = user_message[1:]
        try:
            # Executes the bash command on Termux
            result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            return jsonify({'response': result, 'type': 'terminal'})
        except subprocess.CalledProcessError as e:
            return jsonify({'response': e.output, 'type': 'terminal_error'})
            
    # CHAT HOOK: Standard neural chat
    # (When fully synced, this will ping generate.py)
    return jsonify({
        'response': f"[CRATON CORE SYNTHESIS]: Acknowledged. '{user_message}'\n\n(Note: Mega-Brain is currently resting in Google Drive. This is the local UI simulation bridge.)",
        'type': 'chat'
    })

if __name__ == '__main__':
    print("CRATON DASHBOARD SERVER ONLINE")
    print("Access the Face of Craton at: http://localhost:5000")
    # Run on all local network interfaces so the tablet's browser can easily connect
    app.run(host='0.0.0.0', port=5000, debug=True)
