using System;
using System.Threading.Tasks;

namespace SampleApp.Edge
{
    /// <summary>
    /// Abstract base entity class.
    /// </summary>
    public abstract class BaseEntity
    {
        private static readonly long SerialVersionUID = 1L;
        protected int Id;

        public abstract void Save();

        public int GetId()
        {
            return Id;
        }
    }

    public class OrderService : BaseEntity, IAuditable
    {
        public override void Save()
        {
            // save logic
        }

        public void Audit()
        {
            // audit logic
        }

        public string GetAuditLog()
        {
            return "log";
        }

        public virtual void ProcessOrder(string orderId)
        {
            Save();
            Audit();
        }

        /// <summary>
        /// Processes an order asynchronously.
        /// </summary>
        public async Task<string> ProcessOrderAsync(string orderId)
        {
            await Task.Delay(100);
            return orderId;
        }
    }

    public static class MathHelper
    {
        public static int Add(int a, int b)
        {
            return a + b;
        }

        public static double Multiply(double a, double b)
        {
            return a * b;
        }
    }

    public class Container
    {
        public class InnerConfig
        {
            private int _timeout;
        }

        public void Configure()
        {
            // configuration
        }
    }

    internal class InternalHelper
    {
        internal void DoWork()
        {
            // work
        }
    }

    [TestClass]
    public class OrderServiceTests
    {
        [TestMethod]
        public void TestSaveOrder()
        {
            // test
        }

        [Fact]
        public void ShouldProcessOrder()
        {
            // test
        }

        [Test]
        public void VerifyAuditLog()
        {
            // test
        }
    }
}
