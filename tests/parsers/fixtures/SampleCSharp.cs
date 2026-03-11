using System;
using System.Collections.Generic;

namespace SampleApp.Services
{
    /// <summary>
    /// Service class for managing users.
    /// </summary>
    public class UserService
    {
        private readonly IUserRepository _userRepository;
        private string _serviceName;

        public UserService(IUserRepository userRepository)
        {
            _userRepository = userRepository;
        }

        /// <summary>
        /// Finds a user by their ID.
        /// </summary>
        public User FindById(int id)
        {
            return _userRepository.FindById(id);
        }

        protected void Validate(string input)
        {
            // validation logic
        }
    }

    public interface IAuditable
    {
        void Audit();

        string GetAuditLog();
    }

    public enum OrderStatus
    {
        Pending,
        Completed,
        Cancelled
    }
}
