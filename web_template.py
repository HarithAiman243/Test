css = '''
<style>
.chat-message {
    background-color: #f1f0f0;
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
    max-width: 70%;
}
.chat-message.user {
    background-color: #d1e7dd;
}
.chat-message.avatar {
    width: 40px;
}

.chat-message .avatar img {
    border-radius: 50%;
    max-width: 100%;
    max-height: 40px;
    object-fit: cover;    
}

.chat-message .message {
    margin-left: 10px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
'''

bot_template = '''

<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/2Nf1m3H/AI-avatar.png" alt="Bot Avatar">
    </div>
    <div class="message">
        {message}
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/7QpKsCX/user-avatar.png" alt="User Avatar">
    </div>
    <div class="message">
        {message}
    </div>
'''