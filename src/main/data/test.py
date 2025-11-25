from abc import abstractmethod, ABC
import enum
from itertools import count
from heapq import heappop, heappush, heapify, nlargest, nsmallest

class Side(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class Status(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"

class Sequencer:
    def __init__(self):
        self._counter = count(1)

    def next(self):
        return next(self._counter)
    
class Order(ABC):
    def __init__(self, order_id: str, price: float, sequence_id: int, status: Status = Status.OPEN):
        self.order_id = order_id
        self.price = price
        self.sequence_id = sequence_id
        self.status = status

    def __repr__(self):
        return f"Order(id={self.order_id}, price={self.price}, seq={self.sequence_id}, status={self.status})"

    def __lt__(self, other):
        if self.price == other.price:
            return self.sequence_id < other.sequence_id
        return self.price < other.price
    
    @abstractmethod
    def get_side_price(self):
        raise NotImplementedError("Subclasses must implement get_side_price method")

sequencer_instance = Sequencer()

class Bid(Order):
    def __init__(self, order_id: str, price: float):
        super().__init__(order_id, price, sequencer_instance.next())
        self.side = Side.BUY
    
    def get_side_price(self):
        return -self.price
    
    def __repr__(self):
        return super().__repr__() + f" side={self.side})"

class Ask(Order):
    def __init__(self, order_id: str, price: float):
        super().__init__(order_id, price, sequencer_instance.next())
        self.side = Side.SELL

    def get_side_price(self):
        return self.price
    
    def __repr__(self):
        return super().__repr__() + f" side={self.side})"

class OrderBook:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.order_map = {}
        
    def add_order(self, order: Order):
        if order.order_id in self.order_map:
            raise ValueError(f"Order with id {order.order_id} already exists.")
        if isinstance(order, Bid):
            heappush(self.bids, (order.get_side_price(), order.order_id))
        elif isinstance(order, Ask):
            heappush(self.asks, (order.get_side_price(), order.order_id))
        else:
            raise ValueError("Order must be either Bid or Ask.")
        self.order_map[order.order_id] = order
    
    def remove_order(self, order_id: str):
        if order_id not in self.order_map or self.order_map[order_id].status == Status.CLOSED:
            raise ValueError(f"Order with id {order_id} does not exist.")
        order = self.order_map[order_id]
        order.status = Status.CLOSED
    
    def perform_cleanup(self, heap):
        while heap and self.order_map[heap[0][1]].status == Status.CLOSED:
            heappop(heap)
    
    def update_order(self, order_id: str, new_price: float):
        if order_id not in self.order_map or self.order_map[order_id].status == Status.CLOSED:
            raise ValueError(f"Order with id {order_id} does not exist.")
        order = self.order_map[order_id]
        self.remove_order(order_id)
        if isinstance(order, Bid):
            new_order = Bid(order_id, new_price)
        elif isinstance(order, Ask):
            new_order = Ask(order_id, new_price)
        self.add_order(new_order)
    
    def get_top_k(self, side: Side, k: int = 1):
        if side == Side.BUY:
            self.perform_cleanup(self.bids)
            return [ self.order_map[order_id].price for _, order_id in nsmallest(k, self.bids)]
        elif side == Side.SELL:
            self.perform_cleanup(self.asks)
            return [ self.order_map[order_id].price for _, order_id in nsmallest(k, self.asks)]
        else:
            raise ValueError("Side must be either BUY or SELL.")
        

