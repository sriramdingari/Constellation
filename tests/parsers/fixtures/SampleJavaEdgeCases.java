package com.example.edge;

/**
 * Abstract base class.
 */
public abstract class BaseEntity {

    private static final long serialVersionUID = 1L;
    protected int id;

    public abstract void save();

    public int getId() {
        return id;
    }
}

public class OrderService extends BaseEntity implements Auditable {

    @Override
    public void save() {
        // save logic
    }

    @Override
    public void audit() {
        // audit logic
    }

    @Override
    public String getAuditLog() {
        return "log";
    }

    public void processOrder(String orderId) {
        save();
        audit();
    }

    public void processOrder(String orderId, int quantity) {
        save();
    }
}

@Repository
public class OrderRepository {

    public Order findById(int id) {
        return null;
    }
}

@Component
public class OrderValidator {

    public boolean validate(Order order) {
        return true;
    }
}

public interface Searchable extends Auditable {

    void search(String query);
}

@Test
public class OrderServiceTest {
}

public class Container {

    public static class InnerConfig {

        private int timeout;
    }

    public void configure() {
        // configuration
    }
}

public enum OrderStatus {
    PENDING,
    COMPLETED,
    CANCELLED;

    public boolean isTerminal() {
        return this == COMPLETED || this == CANCELLED;
    }
}

public class AnnotatedController {

    @PostMapping("/orders")
    public void createOrder(Order order) {
    }

    @PutMapping("/orders")
    public void updateOrder(Order order) {
    }

    @DeleteMapping("/orders")
    public void deleteOrder(int id) {
    }

    @RequestMapping("/health")
    public String health() {
        return "ok";
    }

    @ParameterizedTest
    public void testParam() {
    }
}
