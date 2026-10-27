from Simulator.SimulatorCore import AbstractAgent, AgentDecision, Decision, State


class LCAgent(AbstractAgent):
    agent_id = "Simple1"

    def decide(self, decision: Decision, state: State) -> AgentDecision:
        # 1. Load the data from Data/final.csv.
        # 2. Encode features as X with shape (n_samples, n_encoded_features).
        #    Source feature groups per row, in order:
        #    [Buying_Price, Maintenance_Price, No_of_Doors, Person_Capacity,
        #     Size_of_Luggage, Safety, Fuel_Type, Horse_power].
        #    Categorical groups may expand to multiple encoded columns. Exclude
        #    the leading unnamed index column and Car_Acceptability label.
        # 3. Split X and the target into training, validation, and test sets.
        # 4. Initialize a linear classifier.
        # 5. Calculate the baseline probability of a positive vote.
        # 6. Convert the probability into a binary vote.
        if not decision.decision:
            raise ValueError("The current scenario must provide at least one option.")

        return AgentDecision(
            agent_id=self.agent_id,
            decision=decision,
            options=decision.decision,

            
        )