import torch
import torch.nn as nn



class AdaptiveKalmanFilterNN2(nn.Module):
    def __init__(self, observation_features, action_features):
        """
        Args:
            observation_features (int): Dimension of the input (observation + action).
            action_features (int): Dimension of the state to be predicted.
        """
        super().__init__()
        
        self.register_buffer(
            "P", 
            torch.eye(observation_features).unsqueeze(0)
        )
        self.L_Q = nn.Parameter(torch.eye(observation_features) * 0.01, requires_grad=False)
        self.L_R = nn.Parameter(torch.eye(observation_features) * 0.001, requires_grad=False)

        # Parameters for state prediction
        self.A = nn.Parameter(torch.eye(observation_features))  # State transition matrix
        self.B = nn.Parameter(torch.randn(observation_features, action_features))  # Control input matrix
        #self.Q = nn.Parameter(torch.eye(observation_features) * 0.1)  # Process noise covariance
        
        # Parameters for measurement update
        self.H = nn.Parameter(torch.eye(observation_features), requires_grad=False)  # Measurement matrix
        #self.R = nn.Parameter(torch.eye(observation_features) * 0.1)  # Measurement noise covariance

        self.eps = 1e-6

    def forward(self, state_estimate, previous_action, current_action, observation, is_init):
        """
        Args:
            state_estimate (torch.Tensor): Current state tensor of shape (batch_size, state_dim).
            action (torch.Tensor): Input tensor (action) of shape (batch_size, input_dim).
            observation (torch.Tensor): Measurement tensor (ground truth next state) of shape (batch_size, state_dim).
            is_init  (torch.Tensor): Start of episode, reset filter
        Returns:
            updated_state (torch.Tensor): Updated state after Kalman filter correction.
            prediction_error (torch.Tensor): Prediction error for each sample in the batch.
        """
        if len(is_init.shape) > 1: #batch mode. we assume batch dim is time dim, for now.
            updated_states = []
            prediction_errors = []
            for state_estimate_t, previous_action_t, observation_t, is_init_t in zip(
                state_estimate.unbind(0), previous_action.unbind(0), observation.unbind(0), is_init.unbind(0)
            ):
                if torch.any(is_init_t):
                    self.P = torch.eye(self.A.shape[0], device=self.P.device).unsqueeze(0)
                    state_estimate = observation_t
                elif len(updated_states) == 0:
                    state_estimate = state_estimate_t
                    self.P = torch.eye(self.A.shape[0], device=self.P.device).unsqueeze(0)

                P_current = self.P

                Q = self.L_Q @ self.L_Q.T
                R = self.L_R @ self.L_R.T
    
                predicted_state = state_estimate @ self.A.T + previous_action_t @ self.B.T
    
                AP = torch.einsum('bij,bjk->bik', P_current, self.A.unsqueeze(0).transpose(-1, -2))  # (b, s, s)
                P_pred = (self.A @ AP) + Q  # shape: (b, s, s)
    
                innovation = observation_t - (predicted_state @ self.H.T)
                HP = P_pred @ self.H.T  # shape: (b, s, s)
                S = (self.H @ HP) + R   # shape: (b, s, s)

                I_s = torch.eye(S.shape[-1], device=S.device).unsqueeze(0).expand_as(S)
                S_reg = S + self.eps * I_s

                S_inv = torch.inverse(S_reg)  # or torch.linalg.inv(S_reg)
                K = HP @ S_inv  # shape: (b, s, s)

                updated_state = predicted_state + (innovation @ K.transpose(-1, -2))
                updated_state = updated_state.squeeze(0)

                I = torch.eye(S.shape[-1], device=P_pred.device).unsqueeze(0).expand_as(P_pred)
                KH = K @ self.H
                left = (I - KH)
                P_upd = left @ P_pred @ left.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)

                self.P = P_upd

                prediction_error = innovation
                updated_states.append(updated_state)
                prediction_errors.append(prediction_error)
            updated_state = torch.stack(updated_states, dim=0)
            prediction_error = torch.stack(prediction_errors, dim=0)
        else:
            if torch.any(is_init):
                self.P = torch.eye(self.A.shape[0], device=self.P.device).unsqueeze(0)
                state_estimate = observation

            P_current = self.P

            Q = self.L_Q @ self.L_Q.T
            R = self.L_R @ self.L_R.T

            predicted_state = state_estimate @ self.A.T + previous_action @ self.B.T

            AP = torch.einsum('bij,bjk->bik', P_current, self.A.unsqueeze(0).transpose(-1, -2))  # (b, s, s)
            P_pred = (self.A @ AP) + Q  # shape: (b, s, s)

            innovation = observation - (predicted_state @ self.H.T)
            HP = P_pred @ self.H.T  # shape: (b, s, s)
            S = (self.H @ HP) + R   # shape: (b, s, s)

            I_s = torch.eye(S.shape[-1], device=S.device).unsqueeze(0).expand_as(S)
            S_reg = S + self.eps * I_s

            S_inv = torch.inverse(S_reg)  # or torch.linalg.inv(S_reg)
            K = HP @ S_inv  # shape: (b, s, s)

            updated_state = predicted_state + (innovation @ K.transpose(-1, -2))
            updated_state = updated_state.squeeze(0)

            I = torch.eye(S.shape[-1], device=P_pred.device).unsqueeze(0).expand_as(P_pred)
            KH = K @ self.H
            left = (I - KH)
            P_upd = left @ P_pred @ left.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)

            self.P = P_upd
            prediction_error = innovation

        return updated_state, current_action, prediction_error
    
class AdaptiveKalmanFilterNN(nn.Module):
    def __init__(self, observation_features, action_features):
        """
        Args:
            observation_features (int): Dimension of the input (observation + action).
            action_features (int): Dimension of the state to be predicted.
        """
        super().__init__()
        
        # Parameters for state prediction
        self.A = nn.Parameter(torch.eye(observation_features))  # State transition matrix
        self.B = nn.Parameter(torch.randn(observation_features, action_features + observation_features))  # Control input matrix
        #self.Q = nn.Parameter(torch.eye(observation_features) * 0.1)  # Process noise covariance
        
        # Parameters for measurement update
        self.H = nn.Parameter(torch.eye(observation_features))  # Measurement matrix
        #self.R = nn.Parameter(torch.eye(observation_features) * 0.1)  # Measurement noise covariance
        self.L_Q = nn.Parameter(torch.eye(observation_features))
        self.L_R = nn.Parameter(torch.eye(observation_features))
        self.eps = 1e-6

    def forward(self, state_estimate, previous_action, current_action, observation, is_init):
        """
        Args:
            state_estimate (torch.Tensor): Current state tensor of shape (batch_size, state_dim).
            action (torch.Tensor): Input tensor (action) of shape (batch_size, input_dim).
            observation (torch.Tensor): Measurement tensor (ground truth next state) of shape (batch_size, state_dim).
            is_init  (torch.Tensor): Start of episode, reset filter
        Returns:
            updated_state (torch.Tensor): Updated state after Kalman filter correction.
            prediction_error (torch.Tensor): Prediction error for each sample in the batch.
        """
        Q = self.L_Q @ self.L_Q.T
        R = self.L_R @ self.L_R.T

        #extract frame
        input_tensor = torch.cat((state_estimate, previous_action), dim=-1)
        # Predict next state
        predicted_state = state_estimate @ self.A.T + input_tensor @ self.B.T
        
        # Compute prediction error
        prediction_error = observation - predicted_state
        
        # Compute Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
        S = self.H @ Q @ self.H.T + R  # Innovation covariance
        
        I_s = torch.eye(S.shape[0], device=S.device)
        S_reg = S + self.eps * I_s
        
        HQ = self.H @ Q
        K = torch.linalg.solve(S_reg.transpose(-1, -2), HQ.transpose(-1, -2)).transpose(-1, -2)

#        K =Q @ self.H.T @ torch.inverse(S + 1e-6 * torch.eye(S.size(-1)))  # Kalman gain
        
        # Update state: x' = x + K * (measurement - H * x)
        updated_state = predicted_state + prediction_error @ K.transpose(-1, -2)
        
        # Update process noise covariance: P' = (I - K * H) * P
        #I = torch.eye(self.H.size(0), device=state_estimate.device)

        #Q_new = (I - K @ self.H) @ Q @ (I - K @ self.H).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
       # L_Q_new = torch.linalg.cholesky(Q_new + self.eps * I)
        #self.L_Q.data = L_Q_new.detach()

        #self.Q.data = (I - K @ self.H) @ P
        #PHt = P @ self.H.T
        #self.Q.data = (I - K @ self.H) @ P @ (I - K @ self.H).transpose(-1, -2) + K @ self.R @ K.transpose(-1, -2)

        return updated_state, current_action, prediction_error