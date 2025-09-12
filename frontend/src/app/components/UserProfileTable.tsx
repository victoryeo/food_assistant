import { UserIntf } from "@/types/user";

interface UserProfileTableProps {
  userInfo: UserIntf;
  onLogout: () => void;
}

export default function UserProfileTable({ userInfo, onLogout }: UserProfileTableProps) {
  return (
    <div className="overflow-hidden bg-white shadow sm:rounded-lg">
      <table className="min-w-0">
        <tbody>
          <tr>  
            <th>Name</th>
            <th>Avatar</th>
            <th>Action</th>
          </tr>
          <tr>
            <td className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-tl-lg">
              <div className="font-medium text-gray-900">
                  {userInfo.name || 'User'}
              </div>
            </td>
            <td className="px-4 py-2 flex items-center gap-3">
              <div className="font-medium text-gray-900">
                {userInfo.picture ? (
                  <img
                    src={userInfo.picture}
                    alt="Profile"
                    className="rounded-full"
                    style={{ width: '24px', height: '24px' }}
                    onError={(e) => {
                      console.error('Error loading image:', userInfo.picture, e);
                      const target = e.target as HTMLImageElement;
                      target.style.display = 'none';
                    }}
                  />
                ) : (
                  <div className="text-xs text-red-500">No picture</div>
                )}
              </div>
            </td>
            <td className="px-4 py-2 rounded-tr-lg">
              <div className="font-medium text-gray-900">
                <button
                  onClick={onLogout}
                  className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors"
                >
                  Sign out
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
