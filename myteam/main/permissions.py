from rest_framework.permissions import BasePermission
 
class IsAdminAuthenticated(BasePermission):
 
    def has_permission(self, request, view):
    # Ne donnons l’accès qu’aux utilisateurs administrateurs authentifiés
        clients = ["benhachy"]
        client =  request.user.username in clients
        return bool(request.user and request.user.is_authenticated and client)