document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');

    function appendMessage(role, content, type = 'chat') {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        let avatarText = role === 'user' ? 'U' : 'C';
        let contentClass = 'content';
        
        if (type === 'terminal') contentClass += ' terminal-output';
        if (type === 'terminal_error') contentClass += ' terminal-error';

        let displayContent = content;
        if (type === 'chat') {
            displayContent = content.replace(/\n/g, '<br>');
        }

        msgDiv.innerHTML = `
            <div class="avatar">${avatarText}</div>
            <div class="${contentClass}">${displayContent}</div>
        `;
        
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function sendMessage() {
        const text = input.value.trim();
        if (!text) return;

        // Display user message
        appendMessage('user', text);
        input.value = '';

        try {
            // Send to Flask Backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: text })
            });

            const data = await response.json();
            appendMessage('system', data.response, data.type);
        } catch (error) {
            appendMessage('system', 'Error: Neural Bridge to Termux backend severed.', 'terminal_error');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
