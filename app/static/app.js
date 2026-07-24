document.addEventListener('DOMContentLoaded', () => {
    const chatToggleBtn = document.getElementById('chat-toggle-btn');
    const chatWidget = document.getElementById('chat-widget');
    const chatCloseBtn = document.getElementById('chat-close-btn');
    const langToggleBtn = document.getElementById('lang-toggle-btn');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const quickRepliesContainer = document.getElementById('quick-replies-container');
    
    let currentLang = 'EN'; // 'EN' or 'HI'
    
    // Generate random session ID on load
    const generateSessionId = () => {
        return 'sess_' + Math.random().toString(36).substring(2, 15);
    };
    const sessionId = generateSessionId();

    // Toggle Chat Visibility
    const toggleChat = () => {
        chatWidget.classList.toggle('hidden');
        if (!chatWidget.classList.contains('hidden')) {
            chatInput.focus();
        }
    };

    chatToggleBtn.addEventListener('click', toggleChat);
    chatCloseBtn.addEventListener('click', toggleChat);

    // Language Toggle
    langToggleBtn.addEventListener('click', () => {
        currentLang = currentLang === 'EN' ? 'HI' : 'EN';
        langToggleBtn.textContent = currentLang;
        
        // Update welcome message
        document.getElementById('welcome-text-en').style.display = currentLang === 'EN' ? 'block' : 'none';
        document.getElementById('welcome-text-hi').style.display = currentLang === 'HI' ? 'block' : 'none';
        
        // Update input placeholder
        chatInput.placeholder = currentLang === 'EN' ? 'Type your message here...' : 'अपना संदेश यहाँ टाइप करें...';
        
        // Update quick replies text
        const qrButtons = document.querySelectorAll('.quick-reply-btn');
        qrButtons.forEach(btn => {
            btn.textContent = currentLang === 'EN' ? btn.dataset.en : btn.dataset.hi;
        });
    });

    // Auto-scroll to bottom
    const scrollToBottom = () => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    // Add User Message
    const addUserMessage = (text) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message message-appear';
        msgDiv.innerHTML = `<div class="message-content">${escapeHTML(text)}</div>`;
        if (typingIndicator && typingIndicator.parentNode === chatMessages) {
            chatMessages.insertBefore(msgDiv, typingIndicator);
        } else {
            chatMessages.appendChild(msgDiv);
        }
        scrollToBottom();
    };

    // Add Bot Message (clean, no sources)
    const addBotMessage = (text) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message message-appear';
        msgDiv.innerHTML = `<div class="message-content">${formatText(text)}</div>`;
        if (typingIndicator && typingIndicator.parentNode === chatMessages) {
            chatMessages.insertBefore(msgDiv, typingIndicator);
        } else {
            chatMessages.appendChild(msgDiv);
        }
        scrollToBottom();
    };

    // Add Error Message
    const addErrorMessage = () => {
        const errorText = currentLang === 'EN' 
            ? "Sorry, I am having trouble connecting to the server. Please try again later."
            : "क्षमा करें, मुझे सर्वर से जुड़ने में समस्या आ रही है। कृपया बाद में पुनः प्रयास करें।";
        
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message message-appear';
        msgDiv.innerHTML = `<div class="message-content" style="color: var(--accent-red); border-color: var(--accent-red);">${errorText}</div>`;
        if (typingIndicator && typingIndicator.parentNode === chatMessages) {
            chatMessages.insertBefore(msgDiv, typingIndicator);
        } else {
            chatMessages.appendChild(msgDiv);
        }
        scrollToBottom();
    };

    // Show/Hide Typing Indicator
    const setTyping = (isTyping) => {
        if (isTyping) {
            typingIndicator.classList.remove('hidden');
            chatMessages.appendChild(typingIndicator); // move to bottom
            scrollToBottom();
        } else {
            typingIndicator.classList.add('hidden');
        }
    };

    // Handle Send Message
    const sendMessage = async (text) => {
        if (!text.trim()) return;
        
        // Hide quick replies after first message
        if (quickRepliesContainer && !quickRepliesContainer.classList.contains('hidden')) {
            quickRepliesContainer.style.display = 'none';
        }

        addUserMessage(text);
        chatInput.value = '';
        setTyping(true);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: text,
                    session_id: sessionId
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            setTyping(false);
            addBotMessage(data.answer || data.response || 'No response received.');
            
        } catch (error) {
            console.error('Error in chat request:', error);
            setTyping(false);
            addErrorMessage();
        }
    };

    // Event Listeners for Input
    chatSendBtn.addEventListener('click', () => {
        sendMessage(chatInput.value);
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage(chatInput.value);
        }
    });

    // Quick Replies
    document.querySelectorAll('.quick-reply-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const text = currentLang === 'EN' ? e.target.dataset.en : e.target.dataset.hi;
            sendMessage(text);
        });
    });

    // Helper functions
    const escapeHTML = (str) => {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag] || tag)
        );
    };

    const formatText = (text) => {
        // Simple markdown parsing for bold and line breaks
        let formatted = escapeHTML(text);
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\n/g, '<br>');
        return formatted;
    };
});
