// TypeScript namespace
namespace Validators {
  export function isValid(value: string): boolean {
    return value.length > 0;
  }
}

// Test functions (describe/it/test)
describe('UserService', () => {
  it('should fetch user', () => {
    const service = new UserService('http://localhost');
    expect(service).toBeDefined();
  });

  test('should delete user', () => {
    const service = new UserService('http://localhost');
    expect(service).toBeDefined();
  });
});

// Arrow function assigned to const
const calculateTotal = (items: number[]): number => {
  return items.reduce((sum, item) => sum + item, 0);
};

// Export class
export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async get(path: string): Promise<Response> {
    return fetch(`${this.baseUrl}${path}`);
  }

  async post(path: string, data: unknown): Promise<Response> {
    return fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }
}

// Export default function
export default function createClient(url: string): ApiClient {
  return new ApiClient(url);
}

// Async arrow function
const fetchData = async (url: string): Promise<string> => {
  const response = await fetch(url);
  return response.text();
};
