class ChatbotCore:
    def __init__(self):
        self.agents = []
        self.user_profiles = {}
    
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def remove_agent(self, agent):
        self.agents.remove(agent)
    
    def get_agents(self):
        return self.agents
    
    def process_user_input(self, user_input, user_id):
        # Process the user input and delegate the task to the appropriate agent
        for agent in self.agents:
            if agent.can_handle(user_input):
                response = agent.handle(user_input, self.user_profiles.get(user_id, {}))
                return response
        
        # If no agent can handle the user input, return a default response or error message
        return "I'm sorry, I couldn't understand your request."
    
    def get_agent_responses(self, user_input, user_id):
        # Get responses from all agents that can handle the user input
        responses = []
        for agent in self.agents:
            if agent.can_handle(user_input):
                response = agent.handle(user_input, self.user_profiles.get(user_id, {}))
                responses.append(response)
        return responses
    
    def get_all_agent_capabilities(self):
        # Get the capabilities of all agents
        capabilities = []
        for agent in self.agents:
            capabilities.extend(agent.get_capabilities())
        return capabilities
    
    def generate_help_message(self):
        # Generate a help message listing the capabilities of all agents
        help_message = "Here are the things I can help you with:\n"
        capabilities = self.get_all_agent_capabilities()
        for capability in capabilities:
            help_message += f"- {capability}\n"
        return help_message
    
    def get_agent_by_name(self, agent_name):
        # Retrieve an agent by its name
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        return None
    
    def get_agent_by_capability(self, capability):
        # Retrieve an agent that has the specified capability
        for agent in self.agents:
            if capability in agent.get_capabilities():
                return agent
        return None
    
    def get_agent_by_type(self, agent_type):
        # Retrieve agents of a specific type
        agents_of_type = []
        for agent in self.agents:
            if isinstance(agent, agent_type):
                agents_of_type.append(agent)
        return agents_of_type
    
    def register_user_profile(self, user_id, profile):
        # Register a user profile
        self.user_profiles[user_id] = profile
    
    def get_user_profile(self, user_id):
        # Retrieve a user profile
        return self.user_profiles.get(user_id, {})
    
    def log_user_interaction(self, user_id, user_input, agent_name, response):
        # Log user interactions for analytics or monitoring purposes
        # You can customize this method to store or analyze the interaction data
        pass
