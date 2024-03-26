import nltk
from nltk.chat.util import Chat, reflections

# Define the responses for each course or department
responses = {
    'computer science': {
        'style': 'tech_savvy',
        'responses': [
            "Welcome to the cutting-edge world of Computer Science! This field is all about innovation, problem-solving, and pushing the boundaries of technology. What specific area would you like to explore? We offer courses in programming, artificial intelligence, cybersecurity, and more.",
            "Computer Science is the driving force behind the digital revolution. From developing groundbreaking software to designing intelligent systems, the possibilities are endless. Let me know which aspect piques your interest the most, and I'll provide you with all the details you need."
        ]
    },
    'business': {
        'style': 'professional',
        'responses': [
            "Greetings! The Business department is dedicated to preparing you for a successful career in the dynamic world of commerce. We offer comprehensive programs in management, marketing, finance, and entrepreneurship. Which area aligns best with your professional goals?",
            "The business world is constantly evolving, and our department is committed to equipping you with the skills and knowledge to stay ahead of the curve. Please let me know which discipline you'd like to focus on, and I'll provide you with detailed information about our relevant courses and resources."
        ]
    },
    'engineering': {
        'style': 'technical',
        'responses': [
            "Welcome to the realm of Engineering, where innovation meets practicality. Our department offers specialized programs in various disciplines, including civil, mechanical, electrical, and computer engineering. Which branch aligns with your interests and career aspirations?",
            "Engineering is the backbone of technological advancements and infrastructure development. Our department is dedicated to providing you with a solid foundation in theoretical concepts and hands-on practical experience. Please specify the area you'd like to focus on, and I'll provide you with detailed information about our course offerings and state-of-the-art facilities."
        ]
    }
    # Add more courses or departments and responses as needed
}

# Define the chatbot's initial prompt
chatbot_prompt = "Welcome to the university course information chatbot! Please select a course or department from the following options: computer science, business, engineering."

# Define a function to handle user input
def chat_response(user_input):
    user_input = user_input.lower()
    if user_input in responses:
        style = responses[user_input]['style']
        response_list = responses[user_input]['responses']
        return response_list
    else:
        return ["I'm sorry, I don't have information on that course or department. Please select from the available options: computer science, business, engineering."]

# Create a NLTK chatbot
chatbot = Chat(chat_response, reflections)

# Start the chatbot conversation
print(chatbot_prompt)
chatbot.converse()