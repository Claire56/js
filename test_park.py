import pytest
from park import ParkingLot

class TestParkkingLot:
    def test_one_car_added(self):
        parkinglot = ParkingLot()
        parkinglot.park('car')
        assert parkinglot.spaces['car'] == 4

    def test_one_motocycle_added(self):
        parkinglot = ParkingLot()
        parkinglot.park('motorcycle')
        assert parkinglot.spaces['motorcycle'] == 2
    
    def test_one_truck_added(self):
        parkinglot = ParkingLot()
        parkinglot.park('truck')
        # assert parkinglot.spaces['truck'] == 2
        assert parkinglot.spaces.get('truck')==2