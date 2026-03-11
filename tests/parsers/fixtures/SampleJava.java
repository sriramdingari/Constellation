package com.example;

/**
 * Service class for managing users.
 */
@Service
public class UserService {

    private final UserRepository userRepository;
    private String serviceName;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    /**
     * Finds a user by their ID.
     */
    public User findById(int id) {
        return userRepository.findById(id);
    }

    protected void validate(String input) {
        // validation logic
    }
}

/** REST controller for user endpoints. */
@RestController
public class UserController {

    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/users")
    public List<User> getUsers() {
        return userService.findAll();
    }
}

public interface Auditable {

    void audit();

    String getAuditLog();
}
