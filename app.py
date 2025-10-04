from flask import Flask, render_template, request, jsonify, session
import os
import uuid
from datetime import datetime
from gptoss_model import gptoss_model

app = Flask(__name__)
app.secret_key = 'gptoss_chatbot_secret_key_2024'

# Store chat history
chat_sessions = {}

class ChatSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.messages = []
        self.created_at = datetime.now()
    
    def add_message(self, role, content, image_path=None, csv_used=False):
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_path': image_path,
            'csv_used': csv_used
        }
        self.messages.append(message)
        
        # Keep only last 50 messages
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        chat_sessions[session['session_id']] = ChatSession(session['session_id'])
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if 'session_id' not in session:
        return jsonify({'error': 'No session found'}), 400
    
    session_id = session['session_id']
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession(session_id)
    
    chat_session = chat_sessions[session_id]
    
    # Get user input
    user_message = request.form.get('message', '').strip()
    image_file = request.files.get('image')
    use_csv = request.form.get('use_csv', 'false').lower() == 'true'
    
    if not user_message and not image_file:
        return jsonify({'error': 'No message or image provided'}), 400
    
    # Handle image upload
    image_path = None
    if image_file and image_file.filename:
        os.makedirs('static/uploads', exist_ok=True)
        filename = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_file.filename}"
        image_path = os.path.join('static/uploads', filename)
        image_file.save(image_path)
    
    # Add user message to history
    chat_session.add_message('user', user_message, image_path)
    
    try:
        # Generate response using GPT-OSS model
        response = gptoss_model.generate_response(
            text_input=user_message,
            image_path=image_path,
            use_csv=use_csv
        )
        
        # Add assistant response to history
        chat_session.add_message('assistant', response, None, use_csv)
        
        return jsonify({
            'success': True,
            'response': response,
            'session_id': session_id,
            'message_count': len(chat_session.messages),
            'csv_used': use_csv
        })
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        chat_session.add_message('assistant', error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/history')
def get_history():
    if 'session_id' not in session:
        return jsonify({'messages': []})
    
    session_id = session['session_id']
    if session_id not in chat_sessions:
        return jsonify({'messages': []})
    
    chat_session = chat_sessions[session_id]
    return jsonify({'messages': chat_session.messages})

@app.route('/csv_data')
def get_csv_data():
    """Return information about available CSV data"""
    csv_info = []
    for filename, df in gptoss_model.csv_data.items():
        csv_info.append({
            'filename': filename,
            'columns': list(df.columns),
            'total_rows': len(df)
        })
    
    return jsonify({'csv_files': csv_info})

@app.route('/search_csv')
def search_csv():
    """Search through CSV data"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'results': []})
    
    results = gptoss_model.search_csv_data(query)
    return jsonify({'results': results[:10]})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history for current session"""
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in chat_sessions:
            chat_sessions[session_id].messages = []
    
    return jsonify({'success': True})

@app.route('/reload_csv', methods=['POST'])
def reload_csv():
    """Reload CSV files"""
    try:
        gptoss_model.csv_data = gptoss_model.load_csv_data()
        return jsonify({
            'success': True, 
            'message': f'Reloaded {len(gptoss_model.csv_data)} CSV files',
            'files': list(gptoss_model.csv_data.keys())
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 GPT-OSS Flask Chatbot Starting...")
    print("📊 Loaded CSV files:", list(gptoss_model.csv_data.keys()))
    print("🌐 Server running at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)