import React, { useState, useEffect, useRef } from 'react';
import CssBaseline from '@mui/material/CssBaseline';
import Typography from '@mui/joy/Typography';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const lastMessageRef = useRef(null);

  const addMessage = (message) => {
    setIsLoading(true);
    const newMessage = { text: message, answer: '' };
    setMessages((prevMessages) => [...prevMessages, newMessage]);

    startSSE();
  };

  const startSSE = () => {
    const eventSource = new EventSource(`${process.env.REACT_APP_BACKEND_URL}/stream`);

    eventSource.onmessage = (event) => {
      const eventData = JSON.parse(event.data);

      if (eventData.type === 'token') {
        setMessages((prevMessages) =>
          prevMessages.map((msg, index) =>
            index === prevMessages.length - 1
              ? { ...msg, answer: msg.answer + eventData.content + ' ' }
              : msg
          )
        );
      } else if (eventData.type === 'end') {
        setIsLoading(false);
        eventSource.close();
      } else if (eventData.type === 'error') {
        setMessages((prevMessages) =>
          prevMessages.map((msg, index) =>
            index === prevMessages.length - 1 ? { ...msg, answer: 'Error occurred.' } : msg
          )
        );
        setIsLoading(false);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      setMessages((prevMessages) =>
        prevMessages.map((msg, index) =>
          index === prevMessages.length - 1 ? { ...msg, answer: 'Error occurred.' } : msg
        )
      );
      setIsLoading(false);
      eventSource.close();
    };
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const inputField = e.target.elements.question;
    const question = inputField.value.trim();

    if (question === '') {
      console.log("Empty input, not submitting");
      return;
    }

    addMessage(question);
    inputField.value = '';

    fetch(`${process.env.REACT_APP_BACKEND_URL}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    }).catch((error) => {
      console.error('Error starting SSE:', error);
      setMessages((prevMessages) =>
        prevMessages.map((msg, index) =>
          index === prevMessages.length - 1 ? { ...msg, answer: 'Error occurred.' } : msg
        )
      );
      setIsLoading(false);
    });
  };

  useEffect(() => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <>
      <CssBaseline />
      <div className="app-container">
        <form onSubmit={handleSubmit}>
          <input
            name="question"
            type="text"
            placeholder="Enter your question"
            style={{ width: '80%', padding: '0.5rem', marginBottom: '1rem' }}
          />
          <button type="submit" style={{ padding: '0.5rem 1rem' }}>
            {isLoading ? 'Loading...' : 'Submit'}
          </button>
        </form>

        <div className="messages-container">
          {messages.map((msg, index) => (
            <div key={index} ref={lastMessageRef}>
              <Typography level="h2" sx={{ fontSize: '1.8rem', marginBottom: '2rem', fontWeight: 'bold' }}>
                {msg.text}
              </Typography>
              <Typography level="h3" sx={{ fontSize: '1.5rem', marginBottom: '0rem', fontWeight: 'bold' }}>
                Answer:
              </Typography>
              <Typography sx={{ fontSize: '1.2rem', marginBottom: '1rem' }}>
                {msg.answer}
              </Typography>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

export default App;
