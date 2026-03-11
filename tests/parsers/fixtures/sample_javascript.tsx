import React, { useState, useContext } from 'react';
import { ThemeContext } from './ThemeContext';

// TypeScript interface
interface UserProps {
  name: string;
  age: number;
  email?: string;
}

// Class with methods
class UserService {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async fetchUser(id: number): Promise<UserProps> {
    const response = await fetch(`${this.baseUrl}/users/${id}`);
    return response.json();
  }

  deleteUser(id: number): void {
    fetch(`${this.baseUrl}/users/${id}`, { method: 'DELETE' });
  }
}

// Top-level function
function formatUserName(user: UserProps): string {
  return `${user.name} (age: ${user.age})`;
}

// Arrow function component using hooks
const UserCard: React.FC<UserProps> = ({ name, age }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const theme = useContext(ThemeContext);

  return (
    <div className={theme.cardClass}>
      <h2>{name}</h2>
      {isExpanded && <p>Age: {age}</p>}
      <button onClick={() => setIsExpanded(!isExpanded)}>Toggle</button>
    </div>
  );
};

// Named export
export { formatUserName };

// Default export
export default UserCard;
