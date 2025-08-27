class ParkingLot:
    '''
    Basic Parking Lot
Design a parking lot with different vehicle types (car, motorcycle, truck)
Each vehicle type takes different number of spots
Implement park() and unpark() methods
    '''
    def __init__(self):
        self.spaces = {'car': 5, 'motorcycle':3, 'truck': 3}

    def park(self, vehicle):
        if self.spaces[vehicle] > 0:
            if vehicle == 'car':
                self.spaces['car'] -= 1
            elif vehicle == 'motorcycle':
                self.spaces['motorcycle'] -= 1
            elif vehicle == 'truck':
                self.spaces['truck'] -= 1
            else:
                print("Invalid vehicle type")
                return "Invalid vehicle type"
            return "parked successfully"
        else:
            return "sorry no more spaces available"
    def unpark(self, vehicle):
        pass

    def get_available_spots(self, vehicle_type):
        # Return count of available spots
        return self.spaces.get(vehicle_type)

class ParkingLot2:
    '''
    Parking Spot Assignment
Given a parking lot with numbered spots, find the nearest available spot
Return the spot number when parking, or -1 if full
    '''
    def __init__(self, spaces):
        self.spaces = spaces

    def park(self, vehicle):
        pass
    def unpark(self, vehicle):
        pass

# Implement a rate limiter
class RateLimiter:
    def __init__(self, max_requests, window_size):
        pass
    
    def is_allowed(self, user_id):
        # Return True if request is allowed
        pass
    
    def add_request(self, user_id):
        # Record a new request
        pass

# Implement LRU cache
class LRUCache:
    def __init__(self, capacity):
        pass
    
    def get(self, key):
        # Return value or -1 if not found
        pass
    
    def put(self, key, value):
        # Add or update key-value pair
        pass